import matplotlib.pyplot as plt
from fastapi import FastAPI, Request, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from typing import List
import uuid
import os
import aiofiles
import cv2 as cv
import numpy as np
import logging
from zipfile import ZipFile
import img2pdf
import datetime
from PIL import Image
from PIL.ExifTags import TAGS

from src.binarizer.binarizer import Binarizer
from src.binarizer.remove_shadow import RemoveShadow
from src.corner_detector.corner_pipeline import CornerPipeline
from src.enhancer.enhancer import Enhancer
from src.pipeline import Pipeline
from src.warping.warping import Warping

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

TEMP_FOLDER = os.path.join('static', 'sessions')
ALLOWED_IMAGE_TYPES = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}

os.makedirs(TEMP_FOLDER, exist_ok=True)

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


def is_valid_image(filename: str) -> bool:
    """Check if file is a valid image type."""
    return os.path.splitext(filename)[1].lower() in ALLOWED_IMAGE_TYPES


@app.get("/download/zip/{session_id}")
async def download_zip(session_id: str):
    folder_path = os.path.join(TEMP_FOLDER, session_id)
    zip_file_path = os.path.join('static', "zips", f"{session_id}.zip")
    if not os.path.exists(folder_path):
        raise HTTPException(status_code=404, detail="Session folder not found")

    if not os.path.exists(zip_file_path):
        try:
            with ZipFile(zip_file_path, "w") as zip_object:
                for folder, _, filenames in os.walk(folder_path):
                    for filename in filenames:
                        file_path = os.path.join(folder, filename)
                        zip_object.write(file_path, os.path.relpath(file_path, folder_path))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error creating ZIP file: {str(e)}")

    return FileResponse(zip_file_path, media_type='application/zip', filename=f"{session_id}.zip")


def get_exif_time(image_path):
    try:
        img = Image.open(image_path)
        exif_data = img._getexif()

        if exif_data is not None:
            for tag, value in exif_data.items():
                if TAGS.get(tag) == 'DateTime':
                    return value
    except Exception as e:
        print(f"Error reading EXIF data for {image_path}: {e}")
    return None


def sort_images_by_capture_time(folder_path):
    image_files = [
        os.path.join(folder_path, f) for f in os.listdir(folder_path)
        if f.lower().endswith(('jpg', 'jpeg', 'png'))
    ]

    image_files_with_time = []

    for image in image_files:
        exif_time = get_exif_time(image)
        if exif_time:
            try:
                exif_time_obj = datetime.strptime(exif_time, '%Y:%m:%d %H:%M:%S')
                image_files_with_time.append((image, exif_time_obj))
            except ValueError:
                pass

    image_files_with_time.sort(key=lambda x: x[1])
    print(image_files_with_time)

    return [image for image, _ in image_files_with_time]


@app.get("/download/pdf/{session_id}")
async def download_pdf(session_id: str):
    folder_path = os.path.join(TEMP_FOLDER, session_id)
    pdf_file_path = os.path.join('static', 'pdf', f"{session_id}.pdf")

    if not os.path.exists(folder_path):
        raise HTTPException(status_code=404, detail="Session folder not found")

    image_files = sort_images_by_capture_time(folder_path)

    if not image_files:
        image_files = [
            os.path.join(folder_path, f) for f in os.listdir(folder_path)
            if f.lower().endswith(('jpg', 'jpeg', 'png'))
        ]

    try:
        with open(pdf_file_path, "wb") as f:
            f.write(img2pdf.convert(image_files))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating PDF: {str(e)}")

    # Trả về PDF đã tạo
    return FileResponse(pdf_file_path, media_type="application/pdf", filename=f"{session_id}.pdf")


@app.post("/upload-files/")
async def upload_files(
        files: List[UploadFile] = File(...)
):
    try:
        # Validate files
        valid_files = [f for f in files if is_valid_image(f.filename)]

        if not valid_files:
            raise HTTPException(status_code=400, detail="No valid image files uploaded")

        # Generate unique session ID
        session_id = str(uuid.uuid4())
        session_path = os.path.join(TEMP_FOLDER, session_id)
        os.makedirs(session_path, exist_ok=True)

        ok = await process_images(files, session_path)

        if ok:
            return JSONResponse(
                content={
                    "session_id": session_id,
                    "detail": "True",
                    "processed_count": len(valid_files)
                }
            )
        else:
            raise HTTPException(status_code=500, detail="Image not good enough.")

    except Exception as e:
        logger.error(f"Upload error: {e}")
        return JSONResponse(
            content={"error": "Upload failed", "details": str(e)},
            status_code=500
        )


async def process_images(files: List[UploadFile], session_path: str):
    """
    Process uploaded images with error handling and logging.

    Args:
        
        files: List of uploaded files
        session_path: Directory to save processed images
    """
    try:
        for file in files:
            try:
                # Save original file
                file_path = os.path.join(session_path, file.filename)

                content = await file.read()
                np_array = np.frombuffer(content, np.uint8)
                image = cv.imdecode(np_array, cv.IMREAD_COLOR)
                if image is None:
                    logger.warning(f"Could not read image: {file.filename}")
                    continue

                gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

                # Find corners
                corners = CornerPipeline(version="v2").execute(gray)
                if len(corners) == 4:
                    # Warping
                    approx = np.array(corners, dtype=np.float32).reshape((-1, 1, 2))
                    warped_image = Warping(approx).execute_step(image)
                    warped_image = cv.cvtColor(warped_image, cv.COLOR_BGR2GRAY)
                    result = warped_image
                else:
                    result = gray

                # Result
                pipeline = Pipeline(stages=[
                    RemoveShadow(),
                    Enhancer()
                ])

                result = pipeline.execute(result)
                cv.imwrite(file_path, result)

            except Exception as file_error:
                logger.error(f"Error processing {file.filename}: {file_error}")
                continue
        return True

    except Exception as e:
        logger.error("Process failed")
        return False


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Render the main index page."""
    return templates.TemplateResponse(
        request=request,
        name="index.html"
    )


@app.get("/workspace/{session_id}", response_class=HTMLResponse)
async def workspace(request: Request, session_id: str):
    files = os.listdir(os.path.join(TEMP_FOLDER, session_id))
    context = {
        'session_id': session_id,
        'files': files
    }
    return templates.TemplateResponse(request=request, name="workspace.html", context=context)


@app.get("/uploaded-images/{session_id}")
async def get_uploaded_images(session_id: str):
    files = os.listdir(os.path.join(TEMP_FOLDER, session_id))
    print(files)
    context = {
        'session_id': session_id,
        'images': files
    }
    return context


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
