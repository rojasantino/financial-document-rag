import os
from fastapi import APIRouter, UploadFile, File, HTTPException
from app.config import settings
from app.services.ingest import ingest_pdf

router = APIRouter()


@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    os.makedirs(settings.documents_dir, exist_ok=True)
    file_path = os.path.join(settings.documents_dir, file.filename)

    with open(file_path, "wb") as f:
        f.write(await file.read())

    try:
        num_chunks = ingest_pdf(file_path, source_file=file.filename)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    return {
        "message": "Document uploaded and indexed successfully.",
        "filename": file.filename,
        "chunks_indexed": num_chunks,
    }
