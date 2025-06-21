# Budget Normalization

This repository contains a small Python script to normalize credit card statements.

## Setup

```bash
pip install -r requirements.txt
```

The script uses the OpenAI Chat API which requires `openai>=1.24.0`.

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

The script now defaults to the `gpt-4.1-nano` model. Set the `OPENAI_MODEL`
environment variable to override this.

To avoid parse errors the script now requests structured responses from
OpenAI using the `response_format` parameter so the model always returns a
valid JSON object.

The script reads the input CSV as-is without requiring specific column names.
All columns of each row are sent to the Chat API for classification. The
column whose name contains `desc` (case-insensitive) is treated as the
transaction description and will be cleaned using the company name returned by
the model. The output CSV preserves all original columns and appends
`Category` and `Subcategory`.

