"""Utility to normalize credit card statements using OpenAI."""

from __future__ import annotations

import argparse
import json
import logging
import os
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

    date: Optional[str]
    company: Optional[str]
    amount: Optional[float]
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
        "Clean up the merchant name and infer the transaction date, company name, "
        "and amount from these columns. Treat any 'AplPay' or 'Apple Pay' tag as the payment "
        "method, not part of the company name. Remove location references like country or state. "
        "Classify the charge using the categories provided and respond in JSON with keys "
        "'date', 'company', 'amount', 'category', and 'subcategory'.\n\n"
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


def extract_date_amount(row: pd.Series) -> tuple[Optional[str], Optional[float]]:
    """Attempt to infer date and amount values from a row."""
    date_val: Optional[str] = None
    amount_val: Optional[float] = None
    for value in row.values:
        if date_val is None:
            try:
                d = pd.to_datetime(value, errors="coerce")
                if pd.notna(d):
                    date_val = str(d.date())
            except Exception:
                pass
        if amount_val is None:
            try:
                cleaned = str(value).replace("$", "").replace(",", "")
                num = pd.to_numeric(cleaned, errors="coerce")
                if pd.notna(num):
                    amount_val = float(num)
            except Exception:
                pass
        if date_val is not None and amount_val is not None:
            break
    return date_val, amount_val


def infer_company(row: pd.Series) -> str:
    """Guess a description/company field from available columns."""
    for col in row.index:
        name = col.lower()
        if any(key in name for key in ["desc", "merchant", "name"]):
            val = row[col]
            if pd.notna(val):
                return str(val)
    for value in row.values:
        if isinstance(value, str) and value.strip():
            return value
    return " ".join(str(v) for v in row.values if pd.notna(v))


def batch_normalize(df: pd.DataFrame) -> dict[int, TransactionClassification]:
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

        success = False
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
                    data.get("date"),
                    data.get("company"),
                    float(data["amount"]) if "amount" in data and data["amount"] is not None else None,
                    data.get("category", "Uncategorized"),
                    data.get("subcategory", "Uncategorized"),
                )
                success = True
                break
            except Exception as exc:  # broad catch to retry on any failure
                wait = 2 ** attempt
                logger.exception(
                    "Request failed for row %s (attempt %s): %s", idx, attempt + 1, exc
                )
                time.sleep(wait)
        if not success:
            # fall back to simple heuristics when the API fails repeatedly
            desc = infer_company(row)
            date_val, amount_val = extract_date_amount(row)
            cat, sub = categorize(desc)
            results[idx] = TransactionClassification(date_val, desc, amount_val, cat, sub)

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

    results = batch_normalize(df)
    normalized_rows = []
    for idx, row in df.iterrows():
        classification = results.get(idx)
        if classification:
            normalized_rows.append([
                classification.date,
                classification.company,
                classification.amount,
                classification.category,
                classification.subcategory,
            ])
        else:
            desc = infer_company(row)
            date_val, amount_val = extract_date_amount(row)
            cat, sub = categorize(desc)
            normalized_rows.append([date_val, desc, amount_val, cat, sub])

    out = pd.DataFrame(
        normalized_rows,
        columns=["Date", "Company", "Amount", "Category", "Subcategory"],
    )
    out.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
