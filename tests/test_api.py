import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import server  # noqa: E402


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(server, "get_model", lambda: object())
    with TestClient(server.app) as test_client:
        yield test_client


def valid_payload(**changes):
    payload = {
        "idea": "A machine learning system that predicts student dropout risk.",
        "tech_stack": ["Python", "Scikit-learn"],
        "dataset_available": True,
        "compute_level": "Laptop CPU only",
    }
    payload.update(changes)
    return payload


def test_validate_rejects_empty_idea(client):
    response = client.post("/api/validate", json=valid_payload(idea="   "))
    assert response.status_code == 422


def test_validate_rejects_unknown_tech_stack(client):
    response = client.post("/api/validate", json=valid_payload(tech_stack=["UnknownTool"]))
    assert response.status_code == 422


def test_validate_returns_expected_shape(client, monkeypatch):
    monkeypatch.setattr(
        server,
        "compute_novelty_score",
        lambda idea, top_k: {
            "novelty_score": 0.42,
            "interpretation": "Moderately novel",
            "matches": [{
                "title": "Prior work", "abstract": "A study", "source": "github",
                "url": "https://example.com", "tech_stack": "Python", "similarity": 0.58,
            }],
        },
    )
    monkeypatch.setattr(
        server,
        "check_feasibility",
        lambda tech, data, compute, matches: {
            "feasibility_score": 85,
            "verdict": "Feasible",
            "flags": ["No major feasibility risks identified."],
        },
    )
    monkeypatch.setattr(server, "retrieve_products", lambda idea: [{
        "name": "Example", "kind": "website", "category": "Education",
        "description": "Example product", "url": "https://example.com", "similarity": 0.61,
    }])

    response = client.post("/api/validate", json=valid_payload())

    assert response.status_code == 200
    assert response.json() == {
        "novelty": {"score": 0.42, "interpretation": "Moderately novel"},
        "feasibility": {
            "score": 85,
            "verdict": "Feasible",
            "flags": ["No major feasibility risks identified."],
        },
        "matches": [{
            "title": "Prior work", "abstract": "A study", "source": "github",
            "url": "https://example.com", "tech_stack": "Python", "similarity": 0.58,
        }],
        "products": [{
            "name": "Example", "kind": "website", "category": "Education",
            "description": "Example product", "url": "https://example.com", "similarity": 0.61,
        }],
    }


def test_report_requires_api_key(client, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    response = client.post("/api/report", json=valid_payload())
    assert response.status_code == 503
    assert response.json()["detail"] == "Report generation is not configured"
