import argparse
import json
import os

import openai
import pandas as pd

KEYWORD_MAP = {
    'auto insurance': ('Auto & Transport', 'Auto Insurance'),
    'auto payment': ('Auto & Transport', 'Auto Payment'),
    'gas': ('Auto & Transport', 'Gas & Fuel'),
    'fuel': ('Auto & Transport', 'Gas & Fuel'),
    'parking': ('Auto & Transport', 'Parking'),
    'public transportation': ('Auto & Transport', 'Public Transportation'),
    'ride share': ('Auto & Transport', 'Ride Share'),
    'uber': ('Auto & Transport', 'Ride Share'),
    'lyft': ('Auto & Transport', 'Ride Share'),
    'service & parts': ('Auto & Transport', 'Service & Parts'),
    'home phone': ('Bills & Utilities', 'Home Phone'),
    'internet': ('Bills & Utilities', 'Internet'),
    'mobile phone': ('Bills & Utilities', 'Mobile Phone'),
    'television': ('Bills & Utilities', 'Television'),
    'utilities': ('Bills & Utilities', 'Utilities'),
    'books & supplies': ('Education', 'Books & Supplies'),
    'student loan': ('Education', 'Student Loan'),
    'tuition': ('Education', 'Tuition'),
    'amusement': ('Entertainment', 'Amusement'),
    'arts': ('Entertainment', 'Arts'),
    'movies': ('Entertainment', 'Movies & DVDs'),
    'dvd': ('Entertainment', 'Movies & DVDs'),
    'music': ('Entertainment', 'Music'),
    'newspapers': ('Entertainment', 'Newspapers & Magazines'),
    'magazines': ('Entertainment', 'Newspapers & Magazines'),
    'atm fee': ('Fees & Charges', 'ATM Fee'),
    'bank fee': ('Fees & Charges', 'Bank Fee'),
    'finance charge': ('Fees & Charges', 'Finance Charge'),
    'late fee': ('Fees & Charges', 'Late Fee'),
    'service fee': ('Fees & Charges', 'Service Fee'),
    'federal tax': ('Fees & Charges', 'Federal Tax'),
    'local tax': ('Fees & Charges', 'Local Tax'),
    'property tax': ('Fees & Charges', 'Property Tax'),
    'sales tax': ('Fees & Charges', 'Sales Tax'),
    'state tax': ('Fees & Charges', 'State Tax'),
    'trade commission': ('Financial', 'Trade Commissions'),
    'alcohol': ('Food & Dining', 'Alcohol & Bars'),
    'bar': ('Food & Dining', 'Alcohol & Bars'),
    'coffee': ('Food & Dining', 'Coffee Shops'),
    'fast food': ('Food & Dining', 'Fast Food'),
    'delivery': ('Food & Dining', 'Food Delivery'),
    'grocery': ('Food & Dining', 'Groceries'),
    'groceries': ('Food & Dining', 'Groceries'),
    'restaurant': ('Food & Dining', 'Resturants'),
    'dentist': ('Health & Fitness', 'Dentist'),
    'doctor': ('Health & Fitness', 'Doctor'),
    'eyecare': ('Health & Fitness', 'Eyecare'),
    'gym': ('Health & Fitness', 'Gym'),
    'health insurance': ('Health & Fitness', 'Health Insurance'),
    'pharmacy': ('Health & Fitness', 'Pharmacy'),
    'sports': ('Health & Fitness', 'Sports'),
    'furnishings': ('Home', 'Furnishings'),
    'home improvement': ('Home', 'Home Improvement'),
    'home insurance': ('Home', 'Home Insurance'),
    'home services': ('Home', 'Home Services'),
    'home supplies': ('Home', 'Home Supplies'),
    'lawn': ('Home', 'Lawn & Garden'),
    'garden': ('Home', 'Lawn & Garden'),
    'mortgage': ('Home', 'Mortgage & Rent'),
    'rent': ('Home', 'Mortgage & Rent'),
    'bonus': ('Income', 'Bonus'),
    'interest income': ('Income', 'Interest Income'),
    'paycheck': ('Income', 'Paycheck'),
    'reimbursement': ('Income', 'Reimbursement'),
    'rental income': ('Income', 'Rental Income'),
    'hair': ('Personal Care', 'Hair'),
    'laundry': ('Personal Care', 'Laundry'),
    'spa': ('Personal Care', 'Spa & Message'),
    'message': ('Personal Care', 'Spa & Message'),
    'returned purchase': ('Shopping', 'Returned Purchase'),
    'books': ('Shopping', 'Books'),
    'clothing': ('Shopping', 'Clothing'),
    'electronics': ('Shopping', 'Electronics & Software'),
    'software': ('Shopping', 'Electronics & Software'),
    'hobbies': ('Shopping', 'Hobbies'),
    'sporting goods': ('Shopping', 'Sporting Goods'),
    'amazon': ('Shopping', 'Amazon'),
    'air travel': ('Travel', 'Air Travel'),
    'hotel': ('Travel', 'Hotel'),
    'rental car': ('Travel', 'Rental Car & Taxi'),
    'taxi': ('Travel', 'Rental Car & Taxi'),
    'vacation': ('Travel', 'Vacation'),
}

# Build mapping of categories to subcategories for prompt construction
CATEGORIES = {}
for _kw, (cat, subcat) in KEYWORD_MAP.items():
    CATEGORIES.setdefault(cat, set()).add(subcat)


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
    for kw, (cat, subcat) in KEYWORD_MAP.items():
        if kw in desc_low:
            return cat, subcat
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
