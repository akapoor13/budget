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

The input CSV must have the columns `Date`, `Description`, and `Amount`. The script
adds `Category` and `Subcategory` columns based on keyword matching and writes the
output CSV without the original indexes.

