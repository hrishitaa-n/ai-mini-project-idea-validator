"""HTTP API for the AI Mini Project Idea Validator pipeline."""

import os
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from build_database import get_corpus_stats
from feasibility import COMMON_TECH_STACKS, COMPUTE_LEVELS, check_feasibility
from generate_report import generate_report
from products import get_product_count, retrieve_products
from retrieve import compute_novelty_score, get_model


REPORT_LIMIT = 10
REPORT_WINDOW_SECONDS = 60
_report_requests: dict[str, deque[float]] = defaultdict(deque)


class ValidationRequest(BaseModel):
    """Inputs accepted by the validation and report endpoints."""

    idea: str = Field(..., min_length=1)
    tech_stack: list[str]
    dataset_available: bool
    compute_level: str

    @field_validator("idea")
    @classmethod
    def validate_idea(cls, value: str) -> str:
        value = value.strip()
        if not 10 <= len(value) <= 2000:
            raise ValueError("Idea must be between 10 and 2000 characters after trimming")
        return value

    @field_validator("tech_stack")
    @classmethod
    def validate_tech_stack(cls, value: list[str]) -> list[str]:
        unknown = [item for item in value if item not in COMMON_TECH_STACKS]
        if unknown:
            raise ValueError(f"Unknown tech stack item(s): {', '.join(unknown)}")
        return value

    @field_validator("compute_level")
    @classmethod
    def validate_compute_level(cls, value: str) -> str:
        if value not in COMPUTE_LEVELS:
            raise ValueError("Unknown compute level")
        return value


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Retrieval and product search share this singleton model.
    get_model()
    yield


app = FastAPI(title="AI Mini Project Idea Validator API", lifespan=lifespan)


def _pipeline(payload: ValidationRequest) -> tuple[dict, dict, list[dict]]:
    """Run the same pipeline sequence used by the Streamlit app."""
    try:
        novelty_result = compute_novelty_score(payload.idea, top_k=5)
    except FileNotFoundError as error:
        raise HTTPException(
            status_code=503,
            detail=f"Search index is unavailable: {error}",
        ) from error

    feasibility_result = check_feasibility(
        payload.tech_stack,
        payload.dataset_available,
        payload.compute_level,
        novelty_result["matches"],
    )
    products_found = retrieve_products(payload.idea)
    return novelty_result, feasibility_result, products_found


def _public_matches(matches: list[dict]) -> list[dict]:
    """Expose only the documented fields from corpus records."""
    fields = ("title", "abstract", "source", "url", "tech_stack", "similarity")
    return [{field: match.get(field) for field in fields} for match in matches]


def _public_products(products: list[dict]) -> list[dict]:
    """Expose only the documented fields from product catalog records."""
    fields = ("name", "kind", "category", "description", "url", "similarity")
    return [{field: product.get(field) for field in fields} for product in products]


def _allow_report(client_ip: str) -> bool:
    """Allow at most ten report requests per IP in a rolling minute."""
    now = time.monotonic()
    requests = _report_requests[client_ip]
    while requests and now - requests[0] >= REPORT_WINDOW_SECONDS:
        requests.popleft()
    if len(requests) >= REPORT_LIMIT:
        return False
    requests.append(now)
    return True


def _safe_gemini_error(error: Exception) -> str:
    """Return a useful provider error without ever echoing the configured key."""
    message = str(error)
    api_key = os.environ.get("GEMINI_API_KEY")
    if api_key:
        message = message.replace(api_key, "[redacted]")
    return message or "Report generation failed"


@app.get("/api/meta")
def meta() -> dict:
    return {
        "tech_stacks": COMMON_TECH_STACKS,
        "compute_levels": COMPUTE_LEVELS,
        "stats": get_corpus_stats(),
        "product_count": get_product_count(),
        "report_enabled": bool(os.environ.get("GEMINI_API_KEY")),
    }


@app.post("/api/validate")
def validate(payload: ValidationRequest) -> dict:
    novelty_result, feasibility_result, products_found = _pipeline(payload)
    return {
        "novelty": {
            "score": novelty_result["novelty_score"],
            "interpretation": novelty_result["interpretation"],
        },
        "feasibility": {
            "score": feasibility_result["feasibility_score"],
            "verdict": feasibility_result["verdict"],
            "flags": feasibility_result["flags"],
        },
        "matches": _public_matches(novelty_result["matches"]),
        "products": _public_products(products_found),
    }


@app.post("/api/report")
def report(payload: ValidationRequest, request: Request) -> dict:
    if not os.environ.get("GEMINI_API_KEY"):
        raise HTTPException(status_code=503, detail="Report generation is not configured")

    client_ip = request.client.host if request.client else "unknown"
    if not _allow_report(client_ip):
        raise HTTPException(status_code=429, detail="Report limit reached; try again in a minute")

    novelty_result, feasibility_result, products_found = _pipeline(payload)
    try:
        report_text = generate_report(
            payload.idea, novelty_result, feasibility_result, products=products_found
        )
    except Exception as error:
        raise HTTPException(status_code=502, detail=_safe_gemini_error(error)) from error
    return {"report": report_text}


# Mount last so API routes take precedence over the web application's files.
WEB_DIR = Path(__file__).resolve().parent.parent / "web"
app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
