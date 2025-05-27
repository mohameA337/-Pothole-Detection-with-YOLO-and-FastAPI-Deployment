from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from PIL import Image
from ultralytics import YOLO
import io

app = FastAPI()

model = YOLO(r"C:\Users\moham\Documents\GitHub\-Pothole-Detection-with-YOLO-and-FastAPI-Deployment\notebooks\yolov5_project\content\runs\detect\train\weights\best.pt")  


def predict_image(image: Image.Image) -> Image.Image:
    results = model(image)
    annotated_frame = results[0].plot()  
    return Image.fromarray(annotated_frame)

@app.post("/predict")
async def predict_image_route(file: UploadFile = File(...)):
  
    image = Image.open(file.file).convert("RGB")

    predicted_image = predict_image(image)

 
    buffer = io.BytesIO()
    predicted_image.save(buffer, format="PNG")
    buffer.seek(0)

    return StreamingResponse(buffer, media_type="image/png")
