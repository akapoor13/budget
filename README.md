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

The script looks for the OpenAI API key in the `OPENAI_API_KEY` environment
variable. Alternatively you can place the key in a file named
`.openai_api_key` either in the repository root or in your home directory and
it will be loaded automatically.

If the key cannot be located, the script will now raise a clear error
explaining how to provide it.

The script now defaults to the `gpt-4.1-nano` model, which is available for
free tier users. Set the `OPENAI_MODEL` environment variable to override this.

If the OpenAI API returns a rate limit or other transient error, the script
automatically retries the request with exponential backoff using the `backoff`
library.

The input CSV must have the columns `Date`, `Description`, and `Amount`. Each row
is sent to the OpenAI API to clean up the merchant name and determine the most
likely spending category. The output CSV contains the columns `Date`, `Merchant`,
`Amount`, `Category`, and `Subcategory`.

