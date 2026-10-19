import torch
import torch.nn as nn
import numpy as np
import cv2

from pathlib import Path
from PIL import Image

from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from data import CLASS_NAMES


NUM_CLASSES = 12

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

MODEL_PATH = "models/mobilenetv2_finetuned_best.pth"

OUTPUT_DIR = Path("outputs/figures/gradcam")


def create_model():

    weights = MobileNet_V2_Weights.DEFAULT

    model = mobilenet_v2(weights=weights)

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

    model.eval()

    return model


def preprocess_image(image):

    transform = MobileNet_V2_Weights.DEFAULT.transforms()

    input_tensor = transform(image).unsqueeze(0)

    return input_tensor


def find_test_images():

    test_csv = Path("data/processed/test.csv")

    import pandas as pd

    df = pd.read_csv(test_csv)

    # Select one image from different classes
    selected = []

    for class_name in CLASS_NAMES:

        rows = df[df["class"] == class_name]

        if len(rows) > 0:

            selected.append(
                rows.iloc[0]["image_path"]
            )

    return selected[:6]


def generate_gradcam():

    print("Using device:", DEVICE)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    model = create_model().to(DEVICE)

    # Last convolutional layer of MobileNetV2
    target_layers = [
        model.features[-1]
    ]

    cam = GradCAM(
        model=model,
        target_layers=target_layers
    )

    image_paths = find_test_images()

    print(
        f"Generating Grad-CAM for {len(image_paths)} images..."
    )

    for index, image_path in enumerate(image_paths):

        image = Image.open(
            image_path
        ).convert("RGB")

        original = np.array(image)

        original = cv2.resize(
            original,
            (224, 224)
        )

        rgb_image = (
            original.astype(np.float32) / 255.0
        )

        input_tensor = preprocess_image(
            image
        ).to(DEVICE)

        # Model prediction
        with torch.no_grad():

            outputs = model(
                input_tensor
            )

            predicted_class = torch.argmax(
                outputs,
                dim=1
            ).item()

        # Grad-CAM for predicted class
        targets = [
            ClassifierOutputTarget(
                predicted_class
            )
        ]

        grayscale_cam = cam(
            input_tensor=input_tensor,
            targets=targets
        )[0]

        visualization = show_cam_on_image(
            rgb_image,
            grayscale_cam,
            use_rgb=True
        )

        predicted_name = CLASS_NAMES[
            predicted_class
        ]

        output_path = (
            OUTPUT_DIR /
            f"gradcam_{index + 1}_{predicted_name}.jpg"
        )

        Image.fromarray(
            visualization
        ).save(output_path)

        print(
            f"✓ {output_path}"
        )

    print(
        "\n✓ Grad-CAM generation completed"
    )


if __name__ == "__main__":
    generate_gradcam()