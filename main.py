import matplotlib.pyplot as plt
from fastapi import FastAPI, Request, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from typing import List
import uuid
import os
import aiofiles
import cv2 as cv
import numpy as np
import logging

from src.binarizer.binarizer import Binarizer
from src.binarizer.remove_shadow import RemoveShadow
from src.corner_detector.corner_pipeline import CornerPipeline
from src.pipeline import Pipeline
from src.utils import find_top_2_largest_distances
from src.warping.warping import Warping

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
TEMP_FOLDER = os.path.join('static', 'sessions')
ALLOWED_IMAGE_TYPES = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}

# Ensure temp folder exists
os.makedirs(TEMP_FOLDER, exist_ok=True)

app = FastAPI()

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


def is_valid_image(filename: str) -> bool:
    """Check if file is a valid image type."""
    return os.path.splitext(filename)[1].lower() in ALLOWED_IMAGE_TYPES


@app.post("/upload-files/")
async def upload_files(
        files: List[UploadFile] = File(...)
):
    """Handle file uploads and initiate background processing."""
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
                height, width = gray.shape

                # Corner detection
                corners = CornerPipeline(version="v2").execute(gray)
                top_2_distances = find_top_2_largest_distances(corners, width, height)

                vertices = []
                for (point1, point2), _ in top_2_distances:
                    vertices.extend([point1, point2])

                if len(vertices) != 4:
                    logger.warning(f"Could not detect 4 corners in {file.filename}")
                    continue

                # Image processing pipeline
                approx = np.array(vertices, dtype=np.float32).reshape((-1, 1, 2))
                warping_only = Pipeline(stages=[Warping(approx)])
                warped_image = warping_only.execute(image)

                warped_gray = cv.cvtColor(warped_image, cv.COLOR_BGR2GRAY)
                pipeline = Pipeline(stages=[RemoveShadow(), Binarizer()])
                binary = pipeline.execute(warped_gray)
                cv.imwrite(file_path, binary)
            
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
    return templates.TemplateResponse(request=request, name="workspace.html")


@app.get("/uploaded-images/{session_id}")
async def get_uploaded_images(session_id: str):
    files = os.listdir(os.path.join(TEMP_FOLDER, session_id))
    print(files)
    context = {
        'session_id': session_id,
        'images': files
    }
    return context