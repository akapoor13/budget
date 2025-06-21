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

DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-nano")

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


def build_prompt(row: pd.Series) -> str:
    """Construct the user prompt sent to OpenAI using all row columns."""
    row_details = "\n".join(f"{col}: {row[col]}" for col in row.index)
    return (
        "Clean up the merchant name, infer the company, and classify the charge. "
        "Treat any 'AplPay' or 'Apple Pay' tag as the payment method, not part of the company name. "
        "Remove any location, country, city, or state references from the name. "
        "Respond in JSON with keys 'company', 'category', and 'subcategory'.\n\n"
        f"{row_details}\n\nCategories:\n{build_categories_string()}"
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


def batch_normalize(df: pd.DataFrame, desc_col: str) -> dict[int, TransactionClassification]:
    """Normalize rows sequentially using the Chat API with retries."""
    load_api_key()

    results: dict[int, TransactionClassification] = {}
    for idx, row in df.iterrows():
        prompt = build_prompt(row)
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant for personal finance.",
            },
            {"role": "user", "content": prompt},
        ]

        for attempt in range(5):
            try:
                resp = openai.chat.completions.create(
                    model=DEFAULT_MODEL,
                    messages=messages,
                    temperature=0,
                    response_format={"type": "json_object"},
                )
                content = resp.choices[0].message.content
                data = json.loads(content.strip())
                results[idx] = TransactionClassification(
                    data.get("company"),
                    data.get("category", "Uncategorized"),
                    data.get("subcategory", "Uncategorized"),
                )
                break
            except Exception as exc:  # broad catch to retry on any failure
                wait = 2 ** attempt
                logger.exception(
                    "Request failed for row %s (attempt %s): %s", idx, attempt + 1, exc
                )
                time.sleep(wait)
        else:
            desc_val = row.get(desc_col, "")
            cat, sub = categorize(str(desc_val))
            results[idx] = TransactionClassification(None, cat, sub)

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

    desc_col = next((c for c in df.columns if "desc" in c.lower()), df.columns[0])

    results = batch_normalize(df, desc_col)
    normalized_rows = []
    for idx, row in df.iterrows():
        classification = results.get(idx)
        if classification:
            desc = classification.company or row.get(desc_col, "")
            category = classification.category
            subcategory = classification.subcategory
        else:
            desc = row.get(desc_col, "")
            category, subcategory = categorize(str(desc))
        normalized_rows.append([desc, category, subcategory])

    normalized = pd.DataFrame(
        normalized_rows,
        columns=["Description", "Category", "Subcategory"],
    )

    out = df.copy()
    out["Description"] = normalized["Description"]
    out["Category"] = normalized["Category"]
    out["Subcategory"] = normalized["Subcategory"]
    out.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
