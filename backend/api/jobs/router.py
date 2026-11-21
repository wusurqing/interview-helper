from typing import Annotated, Any, Optional
import sqlite3

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from services.storage import job_tracker as jt
from services.storage import review

router = APIRouter()


def _empty_review_summary() -> dict[str, Any]:
    return {
        "review_count": 0,
        "latest_review_id": None,
        "latest_avg_score": None,
        "latest_review_at": None,
        "latest_status": None,
        "linked_review_count": 0,
        "latest_linked_review_id": None,
        "latest_linked_avg_score": None,
        "latest_linked_review_at": None,
        "latest_linked_status": None,
    }


def _attach_review_summaries(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not items:
        return items
    summaries = review.get_application_review_summaries([int(item["id"]) for item in items])
    for item in items:
        item["review_summary"] = summaries.get(int(item["id"]), _empty_review_summary())
    return items


def _attach_review_summary(item: dict[str, Any]) -> dict[str, Any]:
    return _attach_review_summaries([item])[0]


class ApplicationCreate(BaseModel):
    company: str = Field(default="", max_length=200)
    position: str = Field(default="", max_length=200)
    city: str = Field(default="", max_length=100)
    stage: str = Field(default="applied", max_length=64)
    applied_at: Optional[float] = None
    next_followup_at: Optional[float] = None
    interviewer_info: str = Field(default="", max_length=4000)
    feedback: str = Field(default="", max_length=20_000)
    todos: Optional[list[dict[str, Any]]] = Field(default=None, max_length=200)
    notes: str = Field(default="", max_length=50_000)
    sort_order: int = 0


class ApplicationPatch(BaseModel):
    company: Optional[str] = Field(default=None, max_length=200)
    position: Optional[str] = Field(default=None, max_length=200)
    city: Optional[str] = Field(default=None, max_length=100)
    stage: Optional[str] = Field(default=None, max_length=64)
    applied_at: Optional[float] = None
    next_followup_at: Optional[float] = None
    interviewer_info: Optional[str] = Field(default=None, max_length=4000)
    feedback: Optional[str] = Field(default=None, max_length=20_000)
    todos: Optional[list[dict[str, Any]]] = Field(default=None, max_length=200)
    notes: Optional[str] = Field(default=None, max_length=50_000)
    sort_order: Optional[int] = None


class BatchStageBody(BaseModel):
    ids: list[int] = Field(max_length=500)
    stage: str = Field(max_length=64)


class ReorderStageBody(BaseModel):
    """将某阶段下记录的 sort_order 按 ordered_ids 顺序重排（须与该阶段当前 id 集合一致）。"""

    stage: str = Field(max_length=64)
    ordered_ids: list[int] = Field(max_length=500)


class OfferCreate(BaseModel):
    application_id: int
    base_salary: str = Field(default="", max_length=200)
    total_pkg_note: str = Field(default="", max_length=2000)
    bonus: str = Field(default="", max_length=200)
    equity: str = Field(default="", max_length=200)
    benefits: Optional[list[str]] = Field(default=None, max_length=100)
    wfh: str = Field(default="", max_length=200)
    location: str = Field(default="", max_length=200)
    pros: str = Field(default="", max_length=10_000)
    cons: str = Field(default="", max_length=10_000)
    deadline: Optional[float] = None


class OfferPatch(BaseModel):
    base_salary: Optional[str] = Field(default=None, max_length=200)
    total_pkg_note: Optional[str] = Field(default=None, max_length=2000)
    bonus: Optional[str] = Field(default=None, max_length=200)
    equity: Optional[str] = Field(default=None, max_length=200)
    benefits: Optional[list[str]] = Field(default=None, max_length=100)
    wfh: Optional[str] = Field(default=None, max_length=200)
    location: Optional[str] = Field(default=None, max_length=200)
    pros: Optional[str] = Field(default=None, max_length=10_000)
    cons: Optional[str] = Field(default=None, max_length=10_000)
    deadline: Optional[float] = None


class CompareBody(BaseModel):
    offer_ids: list[int] = Field(default_factory=list, max_length=500)


@router.get("/job-tracker/stages")
def api_stages():
    return {"stages": jt.list_stages()}


@router.get("/job-tracker/applications")
def api_list_applications(
    stage: Annotated[Optional[str], Query(max_length=64)] = None,
    q: Annotated[Optional[str], Query(max_length=200)] = None,
    sort_by: Annotated[str, Query(max_length=64)] = "updated_at",
    sort_dir: Annotated[str, Query(max_length=16)] = "desc",
):
    items = jt.list_applications(stage=stage, q=q, sort_by=sort_by, sort_dir=sort_dir)
    return {"items": _attach_review_summaries(items)}


@router.post("/job-tracker/applications")
def api_create_application(body: ApplicationCreate):
    data = body.model_dump(exclude_none=True)
    return _attach_review_summary(jt.create_application(data))


@router.patch("/job-tracker/applications/batch-stage")
def api_batch_stage(body: BatchStageBody):
    n = jt.batch_update_stage(body.ids, body.stage)
    return {"updated": n}


@router.patch("/job-tracker/applications/reorder-stage")
def api_reorder_stage(body: ReorderStageBody):
    n = jt.reorder_stage_applications(body.stage, body.ordered_ids)
    if n <= 0:
        raise HTTPException(400, "Invalid stage or ordered_ids does not match current applications in that stage")
    return {"updated": n}


@router.patch("/job-tracker/applications/{app_id}")
def api_patch_application(app_id: int, body: ApplicationPatch):
    # 保留显式 null，便于清空日期等字段
    patch = body.model_dump(exclude_unset=True)
    if not patch:
        row = jt.get_application(app_id)
        if not row:
            raise HTTPException(404, "Not found")
        return _attach_review_summary(row)
    row = jt.patch_application(app_id, patch)
    if not row:
        raise HTTPException(404, "Not found")
    return _attach_review_summary(row)


@router.get("/job-tracker/applications/{app_id}/reviews")
def api_application_reviews(app_id: int):
    if not jt.get_application(app_id):
        raise HTTPException(404, "Application not found")
    return {"items": review.list_reviews_for_application(app_id)}


@router.delete("/job-tracker/applications/{app_id}")
def api_delete_application(app_id: int):
    if not jt.delete_application(app_id):
        raise HTTPException(404, "Not found")
    return {"ok": True}


@router.get("/job-tracker/offers")
def api_list_offers():
    return {"items": jt.list_offers()}


@router.post("/job-tracker/offers")
def api_create_offer(body: OfferCreate):
    data = body.model_dump(exclude_none=True)
    if not jt.get_application(data["application_id"]):
        raise HTTPException(404, "Application not found")
    try:
        return jt.create_or_update_offer(data)
    except sqlite3.IntegrityError as e:
        raise HTTPException(400, "Invalid application_id") from e


@router.patch("/job-tracker/offers/{offer_id}")
def api_patch_offer(offer_id: int, body: OfferPatch):
    patch = body.model_dump(exclude_unset=True)
    if not patch:
        row = jt.get_offer(offer_id)
        if not row:
            raise HTTPException(404, "Not found")
        return row
    row = jt.patch_offer(offer_id, patch)
    if not row:
        raise HTTPException(404, "Not found")
    return row


@router.delete("/job-tracker/offers/{offer_id}")
def api_delete_offer(offer_id: int):
    if not jt.delete_offer(offer_id):
        raise HTTPException(404, "Not found")
    return {"ok": True}


@router.post("/job-tracker/compare")
def api_compare(body: CompareBody):
    return {"items": jt.compare_offers(body.offer_ids)}
