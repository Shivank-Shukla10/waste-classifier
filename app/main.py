import io
from pathlib import Path
import numpy as np
from PIL import Image
from fastapi.responses import StreamingResponse
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import torch
import torch.nn as nn
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

from src.data import CLASS_NAMES


NUM_CLASSES = 12

MODEL_PATH = Path(
    "models/mobilenetv2_finetuned_best.pth"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# --------------------------------------------------
# FastAPI application
# --------------------------------------------------

app = FastAPI(
    title="Waste Classification API",
    description="Waste image classification using fine-tuned MobileNetV2",
    version="1.0.0"
)


# --------------------------------------------------
# Model
# --------------------------------------------------

def load_model():

    model = mobilenet_v2(
        weights=None
    )

    model.classifier[1] = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(
            model.classifier[1].in_features,
            NUM_CLASSES
        )
    )

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=DEVICE
        )
    )

    model.to(DEVICE)
    model.eval()

    return model


model = load_model()


# --------------------------------------------------
# Preprocessing
# --------------------------------------------------

transform = MobileNet_V2_Weights.DEFAULT.transforms()


# --------------------------------------------------
# Routes
# --------------------------------------------------

@app.get("/")
def root():

    return {
        "message": "Waste Classification API is running",
        "model": "MobileNetV2 Fine-tuned",
        "classes": NUM_CLASSES
    }


@app.get("/health")
def health():

    return {
        "status": "healthy",
        "device": str(DEVICE)
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):

    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a valid image file."
        )

    try:

        contents = await file.read()

        image = Image.open(
            io.BytesIO(contents)
        ).convert("RGB")

    except Exception:

        raise HTTPException(
            status_code=400,
            detail="Unable to read the uploaded image."
        )

    # Preprocess
    input_tensor = transform(
        image
    ).unsqueeze(0).to(DEVICE)

    # Prediction
    with torch.no_grad():

        outputs = model(
            input_tensor
        )

        probabilities = torch.softmax(
            outputs,
            dim=1
        )[0]

    # Top 3 predictions
    top_probabilities, top_indices = torch.topk(
        probabilities,
        k=3
    )

    predictions = []

    for probability, index in zip(
        top_probabilities,
        top_indices
    ):

        predictions.append({
            "class": CLASS_NAMES[index.item()],
            "confidence": round(
                probability.item(),
                4
            )
        })

    return {
        "filename": file.filename,
        "prediction": predictions[0]["class"],
        "confidence": predictions[0]["confidence"],
        "top_3_predictions": predictions
    }


@app.post("/explain")
async def explain(file: UploadFile = File(...)):

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a valid image file."
        )

    contents = await file.read()

    # Open image
    image = Image.open(
        io.BytesIO(contents)
    ).convert("RGB")

    # ---------------------------------------------------------
    # Model input
    # ---------------------------------------------------------

    input_tensor = transform(image).unsqueeze(0)

    # ---------------------------------------------------------
    # Get prediction
    # ---------------------------------------------------------

    with torch.no_grad():
        output = model(input_tensor)

    predicted_class = int(
        torch.argmax(output, dim=1).item()
    )

    # ---------------------------------------------------------
    # Grad-CAM
    # ---------------------------------------------------------

    target_layers = [
        model.features[-1]
    ]

    cam = GradCAM(
        model=model,
        target_layers=target_layers
    )

    targets = [
        ClassifierOutputTarget(predicted_class)
    ]

    grayscale_cam = cam(
        input_tensor=input_tensor,
        targets=targets
    )[0]

    # ---------------------------------------------------------
    # IMPORTANT:
    # Grad-CAM output is 224x224.
    # Resize original image to the same size.
    # ---------------------------------------------------------

    cam_size = (
        grayscale_cam.shape[1],
        grayscale_cam.shape[0]
    )

    rgb_image = np.array(
        image.resize(cam_size)
    ).astype(np.float32) / 255.0

    # ---------------------------------------------------------
    # Create Grad-CAM overlay
    # ---------------------------------------------------------

    visualization = show_cam_on_image(
        rgb_image,
        grayscale_cam,
        use_rgb=True
    )

    # ---------------------------------------------------------
    # Convert to PNG
    # ---------------------------------------------------------

    output_image = Image.fromarray(
        visualization
    )

    image_buffer = io.BytesIO()

    output_image.save(
        image_buffer,
        format="PNG"
    )

    image_buffer.seek(0)

    return StreamingResponse(
        image_buffer,
        media_type="image/png"
    )