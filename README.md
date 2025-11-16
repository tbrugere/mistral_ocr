# Basic CLI for Mistral OCR

I needed to read some pdfs on my phone, but they didn't have text metadata, and were not written for such a small screen. So I wrote this quick CLI tool to extract the text into markdown files using MistralAI's OCR models.

## Installation

using uv

```console
$ uv tool install git+https://github.com/tbrugere/mistral_ocr
```

or using pip directly

```console
$ pip install git+https://github.com/tbrugere/mistral_ocr
```

## Usage

```console
$ mistral-ocr --help
Usage: mistral-ocr [OPTIONS] [FILE_PATH]...

Options:
  --api-key TEXT              API key for Mistral API  [env var:
                              MISTRAL_API_KEY; required]
  --model TEXT
  -o, --output-file FILENAME  save the raw result of the batch job to file
  --resume TEXT               resume from a previously started batch job's id
  --help                      Show this message and exit.
```
