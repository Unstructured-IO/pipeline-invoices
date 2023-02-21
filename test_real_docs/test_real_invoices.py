from prepline_invoices.invoice import (
    DocumentInvoice,
    InvoiceModel,
)

from difflib import get_close_matches

import json
import pytest
import warnings


@pytest.fixture
def invoices():
    files = {}

    # Load the metadata file
    with open("./sample-docs/metadata-30.jsonl") as f:
        for line in f.readlines():
            data = json.loads(line)

            files[data["file_name"]] = json.loads(data["ground_truth"].replace("\\n", " "))[
                "gt_parse"
            ]

    return files


def elements_dict(elements, itemLists):
    d = {e.fieldName: e.text for e in elements}

    itemLines = [{e.fieldName: e.text for e in ielements} for ielements in itemLists]

    if len(itemLines) > 0:
        d["ItemLines"] = [{e.fieldName: e.text for e in ielements} for ielements in itemLists]

    return d


def check_items(dict1, dict2):
    for k1, v1 in dict1.items():
        if k1 in dict2:

            if not get_close_matches(v1, [dict2[k1]], n=1, cutoff=0.9):
                if not get_close_matches(
                    v1.replace(".", "").replace(" ", ""),
                    [dict2[k1].replace(".", "").replace(" ", "")],
                    n=1,
                    cutoff=0.9,
                ):
                    warnings.warn(f"{k1} {v1}/{dict2[k1]}")
        else:
            warnings.warn(f"{k1} not in dict2")


def check(dict1, dict2):
    for k1, v1 in dict1.items():
        assert k1 in dict2

        if k1 != "ItemLines":
            if not get_close_matches(v1, [dict2[k1]], n=1, cutoff=0.9):
                if not get_close_matches(
                    v1.replace(".", "").replace(" ", ""),
                    [dict2[k1].replace(".", "").replace(" ", "")],
                    n=1,
                    cutoff=0.9,
                ):
                    warnings.warn(f"{k1} {v1}/{dict2[k1]}")
        else:
            sorted(v1, key=lambda x: str(x))
            sorted(dict2[k1], key=lambda x: str(x))

            for i in range(len(v1)):
                print("I1", v1[i])
                print("I2", dict2["ItemLines"][i])
                check_items(v1[i], dict2["ItemLines"][i])


def test_invoice(invoices):
    model = InvoiceModel()

    for k, v in invoices.items():
        filename = f"./sample-docs/{k}"

        with open(filename, "rb") as f:
            output = DocumentInvoice.from_file(f, filename, model)
            elements = elements_dict(
                output.__dict__["_pages"][0].elements, output.__dict__["_pages"][0].itemLists
            )

            check(elements, v)
