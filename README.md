# Budget Normalization

This repository contains a small Python script to normalize credit card statements.

## Setup

```bash
pip install -r requirements.txt
```

The script uses the OpenAI Batch API which requires `openai>=1.24.0`.

## Usage

```bash
python normalize_statement.py path/to/statement.csv -o normalized.csv
```

The script looks for the OpenAI API key in the `OPENAI_API_KEY` environment
variable. Alternatively you can place the key in a file named
`.openai_api_key` either in the repository root or in your home directory and
it will be loaded automatically.

If the key cannot be located, the script will now raise a clear error
explaining how to provide it.

The script now defaults to the `gpt-4o-mini` model, which is available for
free tier users. Set the `OPENAI_MODEL` environment variable to override this.

To avoid parse errors the script now requests structured responses from
OpenAI using the `response_format` parameter so the model always returns a
valid JSON object.

The input CSV must have the columns `Date`, `Description`, and `Amount`. All rows
are sent together using the OpenAI Batch API to clean up the merchant name,
infer the company, and determine the most likely spending category. When
available, the inferred company name is used in the `Description` column. The
output CSV contains the columns `Date`, `Description`, `Amount`, `Category`,
and `Subcategory`.

