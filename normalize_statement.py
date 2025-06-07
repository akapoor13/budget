import argparse
import json
import os
import logging
import time
import tempfile

import backoff
import openai
import pandas as pd

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

logger = logging.getLogger(__name__)

# Mapping of available spending categories to their subcategories
KEYWORD_MAP = {
    "Auto & Transport": [
        "Auto Insurance",
        "Auto Payment",
        "Gas & Fuel",
        "Parking",
        "Public Transportation",
        "Ride Share",
        "Service & Parts",
    ],
    "Bills & Utilities": [
        "Home Phone",
        "Internet",
        "Mobile Phone",
        "Television",
        "Utilities",
    ],
    "Education": [
        "Books & Supplies",
        "Student Loan",
        "Tuition",
    ],
    "Entertainment": [
        "Amusement",
        "Arts",
        "Movies & DVDs",
        "Music",
        "Newspapers & Magazines",
    ],
    "Fees & Charges": [
        "ATM Fee",
        "Bank Fee",
        "Finance Charge",
        "Late Fee",
        "Service Fee",
        "Federal Tax",
        "Local Tax",
        "Property Tax",
        "Sales Tax",
        "State Tax",
    ],
    "Financial": [
        "Trade Commissions",
    ],
    "Food & Dining": [
        "Alcohol & Bars",
        "Coffee Shops",
        "Fast Food",
        "Food Delivery",
        "Groceries",
        "Restaurants",
    ],
    "Health & Fitness": [
        "Dentist",
        "Doctor",
        "Eyecare",
        "Gym",
        "Health Insurance",
        "Pharmacy",
        "Sports",
    ],
    "Home": [
        "Furnishings",
        "Home Improvement",
        "Home Insurance",
        "Home Services",
        "Home Supplies",
        "Lawn & Garden",
        "Mortgage & Rent",
    ],
    "Income": [
        "Bonus",
        "Interest Income",
        "Paycheck",
        "Reimbursement",
        "Rental Income",
    ],
    "Personal Care": [
        "Hair",
        "Laundry",
        "Spa & Massage",
    ],
    "Shopping": [
        "Returned Purchase",
        "Books",
        "Clothing",
        "Electronics & Software",
        "Hobbies",
        "Sporting Goods",
        "Amazon",
    ],
    "Travel": [
        "Air Travel",
        "Hotel",
        "Rental Car & Taxi",
        "Vacation",
    ],
}

# Build mapping of categories to subcategories for prompt construction
CATEGORIES = {cat: set(subs) for cat, subs in KEYWORD_MAP.items()}


def load_api_key():
    """Load the OpenAI API key from env or local files."""
    if openai.api_key:
        os.environ.setdefault("OPENAI_API_KEY", openai.api_key)
        return openai.api_key

    key = os.getenv("OPENAI_API_KEY")
    if key:
        openai.api_key = key.strip()
        return openai.api_key

    for path in [".openai_api_key", os.path.expanduser("~/.openai_api_key")]:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as fh:
                openai.api_key = fh.read().strip()
            os.environ["OPENAI_API_KEY"] = openai.api_key
            return openai.api_key

    raise RuntimeError(
        "OpenAI API key not found. Set the OPENAI_API_KEY environment variable or "
        "create a .openai_api_key file."
    )


# The OpenAI Python package moved its error classes from the ``error`` module to
# the package root in v1. To remain compatible with both the 0.x and 1.x
# versions we dynamically pick the location of the error classes.
errors_mod = openai.error if hasattr(openai, "error") else openai


@backoff.on_exception(
    backoff.expo,
    (errors_mod.RateLimitError, errors_mod.APIError),
    max_tries=5,
    logger=logger,
)
def _chat_with_retry(messages):
    """Call OpenAI with exponential backoff on rate limit errors."""
    resp = openai.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0,
        response_format={"type": "json_object"},
    )
    return resp.choices[0].message.content


def openai_normalize(desc: str, date, amount):
    """Use OpenAI to clean merchant name and classify the transaction."""
    load_api_key()

    cats = "\n".join(
        f"- {cat}: {', '.join(sorted(subs))}" for cat, subs in CATEGORIES.items()
    )
    prompt = (
        "Clean up the merchant name and classify the transaction into one of the"
        " following categories and subcategories. Respond in JSON with keys 'clean_"
        "description', 'category', and 'subcategory'.\n\n"
        f"Date: {date}\nDescription: {desc}\nAmount: {amount}\n\nCategories:\n{cats}"
    )
    logger.debug("Prompt sent to OpenAI:\n%s", prompt)
    content = _chat_with_retry(
        [
            {
                "role": "system",
                "content": "You are a helpful assistant for personal finance.",
            },
            {"role": "user", "content": prompt},
        ]
    )
    content = content.strip() if content else ""
    if not content:
        return desc, *categorize(desc)

    try:
        data = json.loads(content)
    except Exception:
        logger.exception("Failed to parse OpenAI response")
        return desc, *categorize(desc)

    return (
        data.get("clean_description", desc),
        data.get("category", "Uncategorized"),
        data.get("subcategory", "Uncategorized"),
    )


def categorize(desc: str):
    desc_low = desc.lower()
    for cat, subcats in CATEGORIES.items():
        for subcat in subcats:
            if subcat.lower() in desc_low:
                return cat, subcat
    for cat in CATEGORIES:
        if cat.lower() in desc_low:
            return cat, "Uncategorized"
    return "Uncategorized", "Uncategorized"


def batch_normalize(df: pd.DataFrame):
    """Normalize all rows at once using the OpenAI Batch API."""
    load_api_key()

    requests = []
    cats = "\n".join(
        f"- {cat}: {', '.join(sorted(subs))}" for cat, subs in CATEGORIES.items()
    )
    for idx, row in df.iterrows():
        prompt = (
            "Clean up the merchant name and classify the transaction into one of the"
            " following categories and subcategories. Respond in JSON with keys 'clean_"
            "description', 'category', and 'subcategory'.\n\n"
            f"Date: {row['Date']}\nDescription: {row['Description']}\nAmount: {row['Amount']}\n\nCategories:\n{cats}"
        )
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant for personal finance.",
            },
            {"role": "user", "content": prompt},
        ]
        requests.append(
            {
                "custom_id": str(idx),
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": MODEL,
                    "messages": messages,
                    "temperature": 0,
                    "response_format": {"type": "json_object"},
                },
            }
        )

    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".jsonl") as fh:
        for req in requests:
            fh.write(json.dumps(req) + "\n")
        input_path = fh.name

    try:
        file_obj = openai.files.create(file=open(input_path, "rb"), purpose="batch")
        batch = openai.batches.create(
            input_file_id=file_obj.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )
        logger.info("Created batch %s", batch.id)
        while True:
            batch = openai.batches.retrieve(batch.id)
            if batch.status in {"completed", "failed", "expired", "cancelled"}:
                break
            time.sleep(5)

        if batch.status != "completed":
            raise RuntimeError(f"Batch finished with status {batch.status}")

        output = openai.files.content(batch.output_file_id).decode("utf-8").splitlines()
    finally:
        os.remove(input_path)

    results = {}
    for line in output:
        rec = json.loads(line)
        cid = int(rec.get("custom_id", 0))
        if "response" in rec and rec["response"].get("status_code") == 200:
            body = json.loads(rec["response"]["body"])
            content = body["choices"][0]["message"]["content"]
            try:
                data = json.loads(content.strip())
                results[cid] = (
                    data.get("clean_description"),
                    data.get("category", "Uncategorized"),
                    data.get("subcategory", "Uncategorized"),
                )
            except Exception:
                logger.exception("Failed to parse response for row %s", cid)
        else:
            logger.error("Request %s failed: %s", cid, rec.get("error"))
    return results


def main():
    parser = argparse.ArgumentParser(description="Normalize a credit card statement")
    parser.add_argument("csvfile", help="Path to input CSV statement")
    parser.add_argument(
        "-o", "--output", help="Output CSV file name", default="normalized.csv"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable debug logging"
    )
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s:%(name)s:%(message)s")

    df = pd.read_csv(args.csvfile)
    for col in ["Date", "Description", "Amount"]:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    df["Date"] = pd.to_datetime(df["Date"]).dt.date

    results = batch_normalize(df)
    normalized_rows = []
    for idx, row in df.iterrows():
        resp = results.get(idx)
        if resp and resp[0]:
            merchant, category, subcategory = resp
        else:
            merchant = row["Description"]
            category, subcategory = categorize(row["Description"])
        normalized_rows.append([merchant, category, subcategory])

    normalized = pd.DataFrame(normalized_rows, columns=["Merchant", "Category", "Subcategory"])

    out = pd.concat([df[["Date", "Amount"]], normalized], axis=1)[
        ["Date", "Merchant", "Amount", "Category", "Subcategory"]
    ]
    out.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
