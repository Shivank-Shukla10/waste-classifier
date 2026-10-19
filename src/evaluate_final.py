import torch
import torch.nn as nn

from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

from data import create_dataloaders, CLASS_NAMES


NUM_CLASSES = 12
BATCH_SIZE = 64
NUM_WORKERS = 2

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def create_model():

    weights = MobileNet_V2_Weights.DEFAULT

    model = mobilenet_v2(weights=weights)

    # Same architecture used during fine-tuning
    model.classifier[1] = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(
            model.classifier[1].in_features,
            NUM_CLASSES
        )
    )

    # Unfreeze the same blocks
    for parameter in model.features.parameters():
        parameter.requires_grad = False

    for block in model.features[-4:]:
        for parameter in block.parameters():
            parameter.requires_grad = True

    for parameter in model.classifier.parameters():
        parameter.requires_grad = True

    checkpoint_path = Path(
        "models/mobilenetv2_finetuned_best.pth"
    )

    model.load_state_dict(
        torch.load(
            checkpoint_path,
            map_location=DEVICE
        )
    )

    print("✓ Loaded final fine-tuned model")

    return model


def evaluate():

    print("Using device:", DEVICE)

    _, _, test_loader = create_dataloaders(
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS
    )

    model = create_model().to(DEVICE)

    model.eval()

    all_predictions = []
    all_labels = []

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(DEVICE)

            outputs = model(images)

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_labels.extend(
                labels.numpy()
            )

    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    macro_f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro"
    )

    print("\n" + "=" * 60)
    print("FINAL TEST RESULTS")
    print("=" * 60)

    print(f"Test Accuracy : {accuracy:.4f}")
    print(f"Test Macro F1 : {macro_f1:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            all_labels,
            all_predictions,
            target_names=CLASS_NAMES,
            digits=4
        )
    )

    cm = confusion_matrix(
        all_labels,
        all_predictions
    )

    print("\nConfusion Matrix:")
    print(cm)

    # Save results
    Path("outputs/predictions").mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        "outputs/predictions/classification_report.txt",
        "w"
    ) as f:

        f.write(
            classification_report(
                all_labels,
                all_predictions,
                target_names=CLASS_NAMES,
                digits=4
            )
        )

    import numpy as np

    np.savetxt(
        "outputs/predictions/confusion_matrix.csv",
        cm,
        delimiter=",",
        fmt="%d"
    )

    print(
        "\n✓ Classification report saved"
    )

    print(
        "✓ Confusion matrix saved"
    )


if __name__ == "__main__":
    evaluate()