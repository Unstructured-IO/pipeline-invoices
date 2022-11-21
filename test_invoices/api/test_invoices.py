from fastapi.testclient import TestClient


from prepline_invoices.api.app import app


def test_section_narrative_api_health_check():
    client = TestClient(app)
    response = client.get("/healthcheck")

    assert response.status_code == 200
