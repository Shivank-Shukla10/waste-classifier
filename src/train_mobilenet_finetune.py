import torch
import torch.nn as nn
import torch.optim as optim

from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

from data import create_dataloaders, get_class_weights


NUM_CLASSES = 12
BATCH_SIZE = 64
EPOCHS = 4
LEARNING_RATE = 1e-5
NUM_WORKERS = 2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_model():
    weights = MobileNet_V2_Weights.DEFAULT
    model = mobilenet_v2(weights=weights)

    # Load the best frozen-backbone model
    checkpoint_path = Path("models/mobilenetv2_frozen_best.pth")

    if checkpoint_path.exists():
        model.classifier[1] = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
        )

        model.load_state_dict(
            torch.load(checkpoint_path, map_location=DEVICE)
        )

        print("✓ Loaded frozen MobileNetV2 checkpoint")

    else:
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    # Freeze everything first
    for parameter in model.features.parameters():
        parameter.requires_grad = False

    # Unfreeze the last few feature blocks
    for block in model.features[-4:]:
        for parameter in block.parameters():
            parameter.requires_grad = True

    # Classifier remains trainable
    for parameter in model.classifier.parameters():
        parameter.requires_grad = True

    return model


def evaluate(model, dataloader, criterion):
    model.eval()

    total_loss = 0.0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = criterion(outputs, labels)
            total_loss += loss.item()

            predictions = torch.argmax(outputs, dim=1)

            all_predictions.extend(
                predictions.cpu().numpy()
            )
            all_labels.extend(
                labels.cpu().numpy()
            )

    avg_loss = total_loss / len(dataloader)

    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    macro_f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro"
    )

    return avg_loss, accuracy, macro_f1


def train():

    print("Using device:", DEVICE)

    train_loader, val_loader, _ = create_dataloaders(
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS
    )

    model = create_model().to(DEVICE)

    class_weights = get_class_weights().to(DEVICE)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    # Only train parameters that were unfrozen
    trainable_parameters = filter(
        lambda p: p.requires_grad,
        model.parameters()
    )

    optimizer = optim.Adam(
        trainable_parameters,
        lr=LEARNING_RATE
    )

    Path("models").mkdir(exist_ok=True)

    best_val_f1 = 0.0

    for epoch in range(EPOCHS):

        model.train()

        running_loss = 0.0

        all_predictions = []
        all_labels = []

        for images, labels in train_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            all_predictions.extend(
                predictions.detach().cpu().numpy()
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

        train_loss = (
            running_loss /
            len(train_loader)
        )

        train_accuracy = accuracy_score(
            all_labels,
            all_predictions
        )

        train_f1 = f1_score(
            all_labels,
            all_predictions,
            average="macro"
        )

        val_loss, val_accuracy, val_f1 = evaluate(
            model,
            val_loader,
            criterion
        )

        print(f"\nEpoch [{epoch + 1}/{EPOCHS}]")

        print(
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy:.4f} | "
            f"Train F1: {train_f1:.4f}"
        )

        print(
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_accuracy:.4f} | "
            f"Val F1: {val_f1:.4f}"
        )

        if val_f1 > best_val_f1:

            best_val_f1 = val_f1

            torch.save(
                model.state_dict(),
                "models/mobilenetv2_finetuned_best.pth"
            )

            print("✓ Best fine-tuned model saved")

    print(
        f"\nBest Validation Macro F1: "
        f"{best_val_f1:.4f}"
    )


if __name__ == "__main__":
    train()