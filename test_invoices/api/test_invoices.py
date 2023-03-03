from fastapi.testclient import TestClient

from glob import glob
import os

from prepline_invoices.api.app import app


def test_section_narrative_api_health_check():
    client = TestClient(app)
    response = client.get("/healthcheck")

    assert response.status_code == 200


def test_invoice_parse_file():
    client = TestClient(app)

    response = client.post(
        "/invoices/v0.0.0/invoices",
        files={
            "files": (
                "sample-docs/4fabfaab-1299.png",
                open("sample-docs/4fabfaab-1299.png", "rb"),
            )
        },
    )
    assert response.status_code == 200


def test_invoice_parse_files():
    client = TestClient(app)

    files = [
        ("files", (file, open(file, "rb")))
        for file in glob("sample-docs/*.png")
        if file != "sample-docs/unstructured_logo.png"
    ]

    response = client.post(
        "/invoices/v0.0.0/invoices",
        files=files,
    )
    assert response.status_code == 200


def test_invoices_api_with_multipart():
    filename = os.path.join("sample-docs/0a7f850d-653.png")
    app.state.limiter.reset()
    client = TestClient(app)
    response = client.post(
        "/invoices/v0.0.0/invoices",
        headers={"Accept": "multipart/mixed"},
        files=[
            ("files", (filename, open(filename, "rb"), "application/pdf")),
            ("files", (filename, open(filename, "rb"), "application/pdf")),
        ],
    )

    assert response.status_code == 200
