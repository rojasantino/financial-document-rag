from fastapi import APIRouter
from pydantic import BaseModel
from app.services.rag_pipeline import answer_question

router = APIRouter()


class Question(BaseModel):
    question: str


@router.post("/ask")
async def ask_question(payload: Question):
    return answer_question(payload.question)
