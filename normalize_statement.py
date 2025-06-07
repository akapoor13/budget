import argparse
import json
import os

import openai
import pandas as pd

# Mapping of available spending categories to their subcategories
KEYWORD_MAP = {
    'Auto & Transport': [
        'Auto Insurance',
        'Auto Payment',
        'Gas & Fuel',
        'Parking',
        'Public Transportation',
        'Ride Share',
        'Service & Parts',
    ],
    'Bills & Utilities': [
        'Home Phone',
        'Internet',
        'Mobile Phone',
        'Television',
        'Utilities',
    ],
    'Education': [
        'Books & Supplies',
        'Student Loan',
        'Tuition',
    ],
    'Entertainment': [
        'Amusement',
        'Arts',
        'Movies & DVDs',
        'Music',
        'Newspapers & Magazines',
    ],
    'Fees & Charges': [
        'ATM Fee',
        'Bank Fee',
        'Finance Charge',
        'Late Fee',
        'Service Fee',
        'Federal Tax',
        'Local Tax',
        'Property Tax',
        'Sales Tax',
        'State Tax',
    ],
    'Financial': [
        'Trade Commissions',
    ],
    'Food & Dining': [
        'Alcohol & Bars',
        'Coffee Shops',
        'Fast Food',
        'Food Delivery',
        'Groceries',
        'Resturants',
    ],
    'Health & Fitness': [
        'Dentist',
        'Doctor',
        'Eyecare',
        'Gym',
        'Health Insurance',
        'Pharmacy',
        'Sports',
    ],
    'Home': [
        'Furnishings',
        'Home Improvement',
        'Home Insurance',
        'Home Services',
        'Home Supplies',
        'Lawn & Garden',
        'Mortgage & Rent',
    ],
    'Income': [
        'Bonus',
        'Interest Income',
        'Paycheck',
        'Reimbursement',
        'Rental Income',
    ],
    'Personal Care': [
        'Hair',
        'Laundry',
        'Spa & Message',
    ],
    'Shopping': [
        'Returned Purchase',
        'Books',
        'Clothing',
        'Electronics & Software',
        'Hobbies',
        'Sporting Goods',
        'Amazon',
    ],
    'Travel': [
        'Air Travel',
        'Hotel',
        'Rental Car & Taxi',
        'Vacation',
    ],
}

# Build mapping of categories to subcategories for prompt construction
CATEGORIES = {cat: set(subs) for cat, subs in KEYWORD_MAP.items()}


def load_api_key():
    """Load the OpenAI API key from env or local files."""
    if openai.api_key:
        return

    key = os.getenv("OPENAI_API_KEY")
    if key:
        openai.api_key = key.strip()
        return

    for path in [".openai_api_key", os.path.expanduser("~/.openai_api_key")]:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as fh:
                openai.api_key = fh.read().strip()
            return


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
    try:
        resp = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant for personal finance.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )
        content = resp["choices"][0]["message"]["content"]
        data = json.loads(content)
        return (
            data.get("clean_description", desc),
            data.get("category", "Uncategorized"),
            data.get("subcategory", "Uncategorized"),
        )
    except Exception:
        return desc, *categorize(desc)

def categorize(desc: str):
    desc_low = desc.lower()
    for cat, subcats in CATEGORIES.items():
        for subcat in subcats:
            if subcat.lower() in desc_low:
                return cat, subcat
    for cat in CATEGORIES:
        if cat.lower() in desc_low:
            return cat, 'Uncategorized'
    return 'Uncategorized', 'Uncategorized'


def main():
    parser = argparse.ArgumentParser(description='Normalize a credit card statement')
    parser.add_argument('csvfile', help='Path to input CSV statement')
    parser.add_argument('-o', '--output', help='Output CSV file name', default='normalized.csv')
    args = parser.parse_args()

    df = pd.read_csv(args.csvfile)
    for col in ['Date', 'Description', 'Amount']:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    df['Date'] = pd.to_datetime(df['Date']).dt.date

    normalized = df.apply(
        lambda row: pd.Series(
            openai_normalize(row['Description'], row['Date'], row['Amount'])
        ),
        axis=1,
    )
    normalized.columns = ['Merchant', 'Category', 'Subcategory']

    out = pd.concat([df[['Date', 'Amount']], normalized], axis=1)[
        ['Date', 'Merchant', 'Amount', 'Category', 'Subcategory']
    ]
    out.to_csv(args.output, index=False)

if __name__ == '__main__':
    main()
