import argparse
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
    df[['Category', 'Subcategory']] = df['Description'].apply(lambda x: pd.Series(categorize(x)))

    out = df[['Date', 'Description', 'Amount', 'Category', 'Subcategory']]
    out.to_csv(args.output, index=False)

if __name__ == '__main__':
    main()
