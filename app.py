from urllib.parse import urlparse

import fitz
import httpx
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, HttpUrl

app = FastAPI(title="HireBeat Resume Parser")
class ParseUrlRequest(BaseModel):
    url: HttpUrl
    filename: str | None = None

@app.get("/health")
def health() -> dict[str, bool]:
    return {"success": True}


@app.post("/parse-pdf")
async def parse_pdf(file: UploadFile = File(...)) -> dict:
    if file.content_type != "application/pdf":
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported in this first version.",
        )

    try:
        file_bytes = await file.read()

        document = fitz.open(
            stream=file_bytes,
            filetype="pdf",
        )

        pages: list[str] = []

        for page in document:
            page_text = page.get_text("text")
            pages.append(page_text)

        document.close()

        text = "\n".join(pages).strip()

        if not text:
            return {
                "success": False,
                "status": "ocr_required",
                "filename": file.filename,
                "text": "",
                "character_count": 0,
                "message": (
                    "No text layer was found. "
                    "This may be a scanned PDF."
                ),
            }

        return {
            "success": True,
            "status": "parsed",
            "filename": file.filename,
            "text": text,
            "character_count": len(text),
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"PDF parsing failed: {error}",
        ) from error
@app.post("/parse-url")
async def parse_url(data: ParseUrlRequest) -> dict:
    url = str(data.url)

    try:
        async with httpx.AsyncClient(
            follow_redirects=True,
            timeout=30.0,
        ) as client:
            response = await client.get(url)

        if response.status_code != 200:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Resume download failed with status "
                    f"{response.status_code}."
                ),
            )

        content_type = (
            response.headers.get("content-type", "")
            .split(";")[0]
            .strip()
            .lower()
        )

        filename = data.filename

        if not filename:
            filename = (
                urlparse(url).path.rsplit("/", 1)[-1]
                or "resume.pdf"
            )

        is_pdf = (
            content_type == "application/pdf"
            or filename.lower().endswith(".pdf")
            or response.content.startswith(b"%PDF")
        )

        if not is_pdf:
            raise HTTPException(
                status_code=400,
                detail=(
                    "The downloaded file does not appear to be a PDF. "
                    f"Content-Type: {content_type or 'unknown'}"
                ),
            )

        document = fitz.open(
            stream=response.content,
            filetype="pdf",
        )

        pages: list[str] = []

        for page in document:
            pages.append(page.get_text("text"))

        page_count = document.page_count
        document.close()

        text = "\n".join(pages).strip()

        if not text:
            return {
                "success": False,
                "status": "ocr_required",
                "filename": filename,
                "content_type": content_type,
                "page_count": page_count,
                "text": "",
                "character_count": 0,
                "message": "No text layer was found in the PDF.",
            }

        return {
            "success": True,
            "status": "parsed",
            "filename": filename,
            "content_type": content_type,
            "page_count": page_count,
            "text": text,
            "character_count": len(text),
        }

    except HTTPException:
        raise

    except httpx.TimeoutException as error:
        raise HTTPException(
            status_code=504,
            detail="Resume download timed out.",
        ) from error

    except httpx.RequestError as error:
        raise HTTPException(
            status_code=502,
            detail=f"Resume download failed: {error}",
        ) from error

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"PDF parsing failed: {error}",
        ) from error