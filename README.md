<h3 align="center">
  <img src="img/unstructured_logo.png" height="200">
</h3>

<h3 align="center">
  <p>Pre-Processing Pipeline Template</p>
</h3>


This repo implements document preprocessing for invoices.

The API is hosted at `https://api.unstructured.io`.

### Updates TODO list

- [x] Update the pipeline name and description in `README.md` (this file)
- [x] Rename all instances of {{ template }} in `README.md` (this file)
- [x] Update the pipeline family name and pipeline package in the `Makefile`
- [x] Update the pipeline name in `preprocessing-pipeline-family.yaml`
- [x] Rename the folders `prepline_template` and `test_template`
- [x] Change name of the variable `PIPELINE_FAMILY` at the top of `.github/workflows/ci.yml`
- [x] If needed, install additional dependencies in the `Dockerfile`
- [x] Add any additional requirements you need to `requirements/base.in` and run `make pip-compile`

## Developer Quick Start

* Using `pyenv` to manage virtualenv's is recommended
	* Mac install instructions. See [here](https://github.com/Unstructured-IO/community#mac--homebrew) for more detailed instructions.
		* `brew install pyenv-virtualenv`
	  * `pyenv install 3.8.13`
  * Linux instructions are available [here](https://github.com/Unstructured-IO/community#linux).

  * Create a virtualenv to work in and activate it, e.g. for one named `invoices`:

	`pyenv  virtualenv 3.8.13 invoices` <br />
	`pyenv activate invoices`

* Run `make install`
* Start a local jupyter notebook server with `make run-jupyter` <br />
	**OR** <br />
	just start the fast-API locally with `make run-web-app`

#### Extracting whatever from some type of document

Give a description of making API calls using example `curl` commands, and example JSON responses.

For example:
```
curl -X 'POST' \
  'http://localhost:8000/pipeline-invoices/v0.0.0/elements' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@need-invoice-name.pdf' | jq -C . | less -R
```

It's also nice to show how to call the API function using pure Python.

### Generating Python files from the pipeline notebooks

You can generate the FastAPI APIs from your pipeline notebooks by running `make generate-api`.

## Security Policy

See our [security policy](https://github.com/Unstructured-IO/pipeline-invoices/security/policy) for
information on how to report security vulnerabilities.

## Learn more

| Section | Description |
|-|-|
| [Company Website](https://unstructured.io) | Unstructured.io product and company info |
