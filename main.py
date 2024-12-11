from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from typing import List
import uuid
import os
import aiofiles

TEMP_FOLDER = "E:\\hus-scanner\\static\\sessions"

if not os.path.exists(TEMP_FOLDER):
    os.mkdir(TEMP_FOLDER)

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")


templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        request=request, name="index.html"
    )


@app.get("/workspace/{session_id}", response_class=HTMLResponse)
async def workspace(request: Request, session_id: str):
    session_folder = os.path.join(TEMP_FOLDER, session_id)
    list_files = os.listdir(session_folder)
    return templates.TemplateResponse(
        request=request, name="workspace.html", context={"session_id": session_id, "files": list_files}
    )


@app.post("/upload-files/")
async def upload_files(files: List[UploadFile] = File(...)):
    try:
        session_id = str(uuid.uuid4())
        session_path = os.path.join(TEMP_FOLDER, session_id)
        os.makedirs(session_path, exist_ok=True)

        for file in files:
            file_path = os.path.join(session_path, file.filename)
            async with aiofiles.open(file_path, 'wb') as f:
                content = await file.read()
                await f.write(content)

        return JSONResponse(content={"session_id": session_id})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/scanned/pdf/{session_id}")
async def download_pdf(session_id: str):
    session_folder = os.path.join(TEMP_FOLDER, session_id)
    list_files = os.listdir(session_folder)
    


