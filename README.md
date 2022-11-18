<h3 align="center">
  <img src="img/unstructured_logo.png" height="200">
</h3>

<h3 align="center">
  <p>Pre-Processing Pipeline for Invoices</p>
</h3>


This repo implements document preprocessing for invoices.

## Developer Quick Start

* Using `pyenv` to manage virtualenv's is recommended
	* Mac install instructions. See [here](https://github.com/Unstructured-IO/community#mac--homebrew) for more detailed instructions.
		* `brew install pyenv-virtualenv`
	  * `pyenv install 3.8.15`
  * Linux instructions are available [here](https://github.com/Unstructured-IO/community#linux).

  * Create a virtualenv to work in and activate it, e.g. for one named `invoices`:

	`pyenv  virtualenv 3.8.15 invoices` <br />
	`pyenv activate invoices`

* Run `make install`
* Start a local jupyter notebook server with `make run-jupyter` <br />
	**OR** <br />
	just start the fast-API locally with `make run-web-app`

#### Extracting Relevant Information from Invoices

To retrieve various elements from an invoice, post the image to the `/invoices` API. You can try this out by starting the API locally with `make run-web-app`. Then from the base folder, run:
```
curl -X 'POST' \
  'http://localhost:8000/invoices/v0.1.0/invoices' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'files=@sample-docs/000d40dd-2812.png' | jq -C . | less -R
```

### Generating Python files from the pipeline notebooks

You can generate the FastAPI APIs from your pipeline notebooks by running `make generate-api`.

## Security Policy

See our [security policy](https://github.com/Unstructured-IO/pipeline-invoices/security/policy) for
information on how to report security vulnerabilities.

## 🤗 Hugging Face

[Hugging Face Spaces](https://huggingface.co/spaces) offer a simple way to host ML demo apps, models and datasets directly on our organization’s profile. This allows us to showcase our projects and work collaboratively with other people in the ML ecosystem. Visit our space [here](https://huggingface.co/unstructuredio)!

## Learn more

| Section | Description |
|-|-|
| [Company Website](https://unstructured.io) | Unstructured.io product and company info |
