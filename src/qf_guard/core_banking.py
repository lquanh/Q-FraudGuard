"""Versioned Core Banking adapter for the Q-FraudGuard event ledger."""

from __future__ import annotations

from hmac import compare_digest
import os
from pathlib import Path
from typing import Annotated, Literal

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, StringConstraints, field_validator

from qf_guard.operations import CaseStore, EventProcessor

MAX_BODY_BYTES = 1_048_576
TOKEN_VARIABLES = {
    "ingest": "QFG_CORE_INGEST_TOKEN",
    "receipts": "QFG_CORE_RECEIPTS_TOKEN",
    "read": "QFG_CORE_READ_TOKEN",
}
Identifier = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=128)]
Amount = Annotated[float, Field(strict=True, ge=0, le=1e15)]
Currency = Annotated[str, StringConstraints(pattern=r"^[A-Z]{3}$")]


class Transaction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    transaction_id: Identifier
    timestamp: AwareDatetime
    sender_id: Identifier
    receiver_id: Identifier
    amount: Amount
    amount_currency: Currency
    type: Identifier = "TRANSFER"
    payment_format: Identifier | None = None
    device_id: Identifier | None = None
    country: Annotated[str, StringConstraints(pattern=r"^[A-Z]{2}$")] | None = None
    old_balance_sender: Amount | None = None
    new_balance_sender: Amount | None = None
    old_balance_receiver: Amount | None = None
    new_balance_receiver: Amount | None = None

    @field_validator("timestamp", mode="before")
    @classmethod
    def timestamp_as_string(cls, value):
        if not isinstance(value, str):
            raise ValueError("timestamp must be an ISO 8601 timestamp with timezone")
        return value

    @field_validator("timestamp")
    @classmethod
    def timestamp_in_supported_range(cls, value):
        pd.Timestamp(value).as_unit("ns")
        return value


class TransactionBatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    transactions: Annotated[list[Transaction], Field(min_length=1, max_length=256)]


class VerificationReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid")

    receipt_id: Identifier
    transaction_id: Identifier
    account_id: Identifier
    expected_version: Annotated[int, Field(strict=True, ge=1, le=2**63 - 1)]
    action: Literal["info_received", "verify_ok", "verify_failed"]
    reason: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)]


class IngestResult(BaseModel):
    received: int
    inserted: int
    duplicates: int


class CaseSnapshot(BaseModel):
    transaction_id: str
    risk_score: float
    graph_score: float
    graph_model: Literal["heuristic", "tgnn"]
    is_candidate: bool
    state: str | None
    version: int | None


class CaseEvent(BaseModel):
    id: int
    stream: str
    transaction_id: str
    account: str
    action: str
    actor: str
    at: str
    previous: str | None
    state: str
    reason: str
    current_version: int
    current_state: str


class CaseEventPage(BaseModel):
    events: list[CaseEvent]
    next_after: int


class ReceiptResult(BaseModel):
    recorded: bool


class BodySizeLimit:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        header = dict(scope["headers"]).get(b"content-length", b"0")
        if not header.isdigit() or len(header) > 7 or int(header) > MAX_BODY_BYTES:
            await JSONResponse({"detail": "Request body exceeds the allowed size."}, 413)(scope, receive, send)
            return
        seen = 0

        async def limited_receive():
            nonlocal seen
            message = await receive()
            seen += len(message.get("body", b""))
            if seen > MAX_BODY_BYTES:
                raise HTTPException(413, "Request body exceeds the allowed size.")
            return message

        await self.app(scope, limited_receive, send)


def _load_processor() -> EventProcessor:
    try:
        database, stream, model = (os.environ[key] for key in ("QFG_CASE_DB", "QFG_STREAM", "QFG_MODEL_PATH"))
    except KeyError as exc:
        raise RuntimeError(f"Set {exc.args[0]} to enable the Core Banking adapter.") from exc
    if not all(value.strip() for value in (database, stream, model)):
        raise RuntimeError("Database, stream and model settings must be nonempty.")
    return EventProcessor.load(CaseStore(Path(database), stream), Path(model))


def create_app(*, processor: EventProcessor | None = None, tokens: dict[str, str] | None = None) -> FastAPI:
    credentials = (tokens if tokens is not None else
                   {role: os.environ.get(variable, "") for role, variable in TOKEN_VARIABLES.items()})
    if set(credentials) != set(TOKEN_VARIABLES) or len(set(credentials.values())) != 3 or any(
        len(value) < 32 or any(not 33 <= ord(char) <= 126 for char in value) for value in credentials.values()
    ):
        raise RuntimeError("Configure three distinct printable Core Banking tokens of at least 32 characters.")
    keys = {role: value.encode("ascii") for role, value in credentials.items()}
    processor = processor or _load_processor()
    store = processor.store
    app = FastAPI(title="Q-FraudGuard Core Banking", version="1.0.0")
    app.add_middleware(BodySizeLimit)
    bearer = HTTPBearer(auto_error=False)

    @app.middleware("http")
    async def no_cache(request: Request, call_next):
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        return response

    def require(role: str):
        def authenticate(auth: HTTPAuthorizationCredentials | None = Depends(bearer)):
            if auth is None or not compare_digest(auth.credentials.encode("utf-8"), keys[role]):
                raise HTTPException(401, "Invalid credentials for this operation.",
                                    headers={"WWW-Authenticate": "Bearer"})
        return Depends(authenticate)

    @app.exception_handler(RequestValidationError)
    async def invalid_request(_request: Request, _error: RequestValidationError):
        return JSONResponse({"detail": "Request does not match the Core Banking contract."}, 422)

    @app.get("/healthz")
    def health():
        with store.connection() as db:
            ready = db.execute("SELECT 1 FROM streams WHERE name=?", (store.stream,)).fetchone()
        if ready is None:
            raise HTTPException(503, "Scoring stream unavailable.")
        return {"status": "ready"}

    @app.post("/v1/transactions", dependencies=[require("ingest")], response_model=IngestResult)
    def ingest(batch: TransactionBatch):
        events = [item.model_dump(mode="json", exclude_none=True) for item in batch.transactions]
        try:
            inserted = processor.ingest(pd.DataFrame(events))
        except ValueError as exc:
            message = str(exc)
            if message.startswith(("conflicting payload", "event timestamp precedes")):
                raise HTTPException(409, "Transaction ID conflict or event precedes the committed stream.") from exc
            if message == "events in a batch must be in chronological source order":
                raise HTTPException(422, "Send transactions in chronological source order.") from exc
            raise
        return {"received": len(events), "inserted": inserted, "duplicates": len(events) - inserted}

    @app.get("/v1/cases", dependencies=[require("read")], response_model=CaseSnapshot)
    def read_case(transaction_id: Annotated[Identifier, Query()]):
        row = store.case(transaction_id)
        if row is None:
            raise HTTPException(404, "Transaction not found.")
        return {
            "transaction_id": transaction_id,
            "risk_score": row["risk_score"],
            "graph_score": row["graph_score"],
            "graph_model": row.get("graph_model", "heuristic"),
            "is_candidate": row["is_candidate"],
            "state": row.get("work_state"),
            "version": row.get("work_version"),
        }

    @app.get("/v1/case-events", dependencies=[require("read")], response_model=CaseEventPage)
    def read_events(
        after: Annotated[int, Query(ge=0, le=2**63 - 1)] = 0,
        limit: Annotated[int, Query(ge=1, le=1000)] = 250,
        transaction_id: Annotated[Identifier | None, Query()] = None,
    ):
        changes = store.case_changes(after, limit, txid=transaction_id)
        return {"events": changes, "next_after": changes[-1]["id"] if changes else after}

    @app.post("/v1/verification-receipts", dependencies=[require("receipts")], response_model=ReceiptResult)
    def record_receipt(receipt: VerificationReceipt):
        try:
            recorded = store.customer_response(
                receipt.transaction_id, receipt.action, receipt.reason, receipt.expected_version,
                account=receipt.account_id, receipt_id=receipt.receipt_id,
            )
        except ValueError as exc:
            raise HTTPException(409, "Receipt ID reused with a different payload.") from exc
        if not recorded:
            raise HTTPException(409, "Case version, account or response action does not match an open request.")
        return {"recorded": True}

    return app
