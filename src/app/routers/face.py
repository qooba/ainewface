from fastapi import APIRouter, File, UploadFile
from common import Injects
from services.face import FaceService
from starlette.responses import StreamingResponse

import io

import numpy as np
from PIL import Image


router = APIRouter()

@router.get("/")
async def root(face_service: FaceService = Injects(FaceService)):
    file_processed=face_service.generate()
    bio = io.BytesIO()
    file_processed.save(bio,"JPEG")
    return StreamingResponse(io.BytesIO(bio.getbuffer()), media_type="image/jpeg")

