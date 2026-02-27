import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
import tempfile
import shutil
import requests
from typing import Optional

from inference import get_detector, ONNXYOLODetector

app = FastAPI(title="Vineyard CV Service")

@app.on_event("startup")
async def startup_event():
    try:
        get_detector()
    except Exception as e:
        pass

@app.get("/")
async def root():
    detector = get_detector()
    return {
        "service": "Vineyard CV Service",
        "status": "running",
        "model_loaded": detector is not None,
        "classes": detector.class_names
    }

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.post("/process_video")
async def process_video(
    video_id: int = Form(...),
    video_file: UploadFile = File(...),
    callback_url: Optional[str] = Form(None),
    frame_interval: int = Form(5)
):
    temp_path = None
    
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
            shutil.copyfileobj(video_file.file, tmp)
            temp_path = tmp.name
        
        detector = get_detector()
        results = detector.process_video(temp_path, frame_interval)
        
        result = {
            "bushes_count": results["statistics"]["bushes_count"],
            "gaps_count": results["statistics"]["gaps_count"],
            "bush_spacing_avg": results["statistics"]["bush_spacing_avg"],
            "row_spacing": results["statistics"]["row_spacing"],
            "result_json": {
                "video_info": results["video_info"],
                "tracking_stats": results["tracking_stats"],
                "details": results["statistics"]["details"],
                "bushes_positions": results.get("bushes_positions", []),
                "gaps_positions": results.get("gaps_positions", [])
            }
        }
        
        if callback_url:
            response = requests.post(callback_url, json=result)
            response.raise_for_status()
        
        return {
            "status": "success",
            "video_id": video_id,
            "result": result
        }
        
    except Exception as e:
        if callback_url:
            try:
                requests.post(callback_url, json={"error": str(e)})
            except:
                pass
        
        raise HTTPException(status_code=500, detail=str(e))
        
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)

@app.get("/model_info")
async def model_info():
    detector = get_detector()
    return {
        "model_path": "models/best.onnx",
        "classes": detector.class_names,
        "num_classes": len(detector.class_names),
        "input_size": f"{detector.input_width}x{detector.input_height}",
        "conf_threshold": detector.conf_threshold,
        "iou_threshold": detector.iou_threshold
    }

@app.post("/process_video_sync")
async def process_video_sync(
    video_file: UploadFile = File(...),
    frame_interval: int = Form(5)
):
    temp_path = None
    
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
            shutil.copyfileobj(video_file.file, tmp)
            temp_path = tmp.name
        
        detector = get_detector()
        results = detector.process_video(temp_path, frame_interval)
        
        return {
            "bushes_count": results["statistics"]["bushes_count"],
            "gaps_count": results["statistics"]["gaps_count"],
            "bush_spacing_avg": results["statistics"]["bush_spacing_avg"],
            "row_spacing": results["statistics"]["row_spacing"],
            "video_info": results["video_info"]
        }
        
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)