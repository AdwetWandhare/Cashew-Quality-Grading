from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from model import LABELS_PATH, MODEL_PATH, load_cashew_model
from processor import predict_grade


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model_service = load_cashew_model(MODEL_PATH)
    yield


app = FastAPI(
    title="Automated Quality Grading of Cashew Kernels API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> dict:
    model_service = app.state.model_service
    return {
        "status": "ok",
        "model_path": str(MODEL_PATH),
        "labels_path": str(LABELS_PATH),
        "model_loaded": model_service.weights_loaded,
        "classes": model_service.class_names,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image uploads are supported.")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        prediction = predict_grade(image_bytes=image_bytes, model_service=app.state.model_service)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return prediction
