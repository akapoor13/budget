# Budget Normalization

This repository contains a small Python script to normalize credit card statements.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
python normalize_statement.py path/to/statement.csv -o normalized.csv
```

The input CSV must have the columns `Date`, `Description`, and `Amount`. Each row
is sent to the OpenAI API to clean up the merchant name and determine the most
likely spending category. The output CSV contains the columns `Date`, `Merchant`,
`Amount`, `Category`, and `Subcategory`.

