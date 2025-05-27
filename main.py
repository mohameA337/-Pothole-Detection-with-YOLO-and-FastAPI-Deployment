from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from PIL import Image
from ultralytics import YOLO
import io

app = FastAPI()

model_nano = YOLO(r"C:\Users\moham\Documents\GitHub\-Pothole-Detection-with-YOLO-and-FastAPI-Deployment\notebooks\yolov5_project (1)\content\runs\detect\train\weights\best.pt")
model_small = YOLO(r"C:\Users\moham\Documents\GitHub\-Pothole-Detection-with-YOLO-and-FastAPI-Deployment\notebooks\yolov5_project (1)\content\runs\detect\train2\weights\best.pt")

def predict_and_annotate(model, image: Image.Image) -> Image.Image:
    results = model(image)
    annotated = results[0].plot()  
    return Image.fromarray(annotated)

@app.post("/predict/nano")
async def predict_nano(file: UploadFile = File(...)):
    image = Image.open(file.file).convert("RGB")
    predicted = predict_and_annotate(model_nano, image)

    buffer = io.BytesIO()
    predicted.save(buffer, format="PNG")
    buffer.seek(0)
    return StreamingResponse(buffer, media_type="image/png")

@app.post("/predict/small")
async def predict_small(file: UploadFile = File(...)):
    image = Image.open(file.file).convert("RGB")
    predicted = predict_and_annotate(model_small, image)

    buffer = io.BytesIO()
    predicted.save(buffer, format="PNG")
    buffer.seek(0)
    return StreamingResponse(buffer, media_type="image/png")
