"""Utility to normalize credit card statements using OpenAI."""

from __future__ import annotations

import argparse
import json
import logging
import os
import tempfile
import time
from dataclasses import dataclass
from typing import Optional
import openai
import pandas as pd

DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

logger = logging.getLogger(__name__)


@dataclass
class TransactionClassification:
    """Structured result describing a classified transaction."""

    company: Optional[str]
    category: str
    subcategory: str

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
        "Resturants",
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
        "Spa & Message",
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


def build_categories_string() -> str:
    """Return a formatted list of available categories."""
    return "\n".join(
        f"- {cat}: {', '.join(sorted(subs))}" for cat, subs in CATEGORIES.items()
    )


def build_prompt(desc: str, date, amount) -> str:
    """Construct the user prompt sent to OpenAI."""
    return (
        "Clean up the merchant name, try to infer the company behind the charge,"
        " and classify the transaction into one of the following categories and"
        " subcategories. Respond in JSON with keys 'company', 'category', and"
        " 'subcategory'.\n\n"
        f"Date: {date}\nDescription: {desc}\nAmount: {amount}\n\nCategories:\n{build_categories_string()}"
    )


def load_api_key():
    """Load the OpenAI API key from env or local files."""
    if openai.api_key:
        os.environ.setdefault("OPENAI_API_KEY", openai.api_key)
        logger.debug("Loaded API key from openai.api_key")
        return openai.api_key

    key = os.getenv("OPENAI_API_KEY")
    if key:
        openai.api_key = key.strip()
        logger.debug("Loaded API key from environment variable")
        return openai.api_key

    for path in [".openai_api_key", os.path.expanduser("~/.openai_api_key")]:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as fh:
                openai.api_key = fh.read().strip()
            os.environ["OPENAI_API_KEY"] = openai.api_key
            logger.debug("Loaded API key from %s", path)
            return openai.api_key

    raise RuntimeError(
        "OpenAI API key not found. Set the OPENAI_API_KEY environment variable or "
        "create a .openai_api_key file."
    )



def categorize(desc: str) -> tuple[str, str]:
    """Infer category and subcategory from description using keyword matching."""
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

    requests: list[dict] = []
    for idx, row in df.iterrows():
        prompt = build_prompt(row['Description'], row['Date'], row['Amount'])
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
                    "model": DEFAULT_MODEL,
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

    results: dict[int, TransactionClassification] = {}
    for line in output:
        rec = json.loads(line)
        cid = int(rec.get("custom_id", 0))
        if "response" in rec and rec["response"].get("status_code") == 200:
            body = json.loads(rec["response"]["body"])
            content = body["choices"][0]["message"]["content"]
            try:
                data = json.loads(content.strip())
                results[cid] = TransactionClassification(
                    data.get("company"),
                    data.get("category", "Uncategorized"),
                    data.get("subcategory", "Uncategorized"),
                )
            except Exception:
                logger.exception("Failed to parse response for row %s", cid)
                cat, sub = categorize(df.loc[cid, "Description"])
                results[cid] = TransactionClassification(None, cat, sub)
        else:
            logger.error("Request %s failed: %s", cid, rec.get("error"))
            cat, sub = categorize(df.loc[cid, "Description"])
            results[cid] = TransactionClassification(None, cat, sub)
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
        classification = results.get(idx)
        if classification:
            desc = classification.company or row["Description"]
            category = classification.category
            subcategory = classification.subcategory
        else:
            desc = row["Description"]
            category, subcategory = categorize(row["Description"])
        normalized_rows.append([desc, category, subcategory])

    normalized = pd.DataFrame(
        normalized_rows,
        columns=["Description", "Category", "Subcategory"],
    )

    out = pd.DataFrame(
        {
            "Date": df["Date"],
            "Description": normalized["Description"],
            "Amount": df["Amount"],
            "Category": normalized["Category"],
            "Subcategory": normalized["Subcategory"],
        }
    )
    out.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
