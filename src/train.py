import torch
import torch.nn as nn
import torch.optim as optim

from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score

from data import (
    create_dataloaders,
    get_class_weights
)

torch.set_num_threads(4)

# -----------------------------
# Configuration
# -----------------------------

NUM_CLASSES = 12
BATCH_SIZE = 64
EPOCHS = 5
LEARNING_RATE = 0.001

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Device will be printed inside train()


# -----------------------------
# Baseline CNN
# -----------------------------

class BaselineCNN(nn.Module):

    def __init__(self, num_classes=NUM_CLASSES):

        super().__init__()

        self.features = nn.Sequential(

            # Block 1
            nn.Conv2d(
                3, 32,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 2
            nn.Conv2d(
                32, 64,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 3
            nn.Conv2d(
                64, 128,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(

            nn.AdaptiveAvgPool2d((1, 1)),

            nn.Flatten(),

            nn.Linear(128, 128),

            nn.ReLU(),

            nn.Dropout(0.4),

            nn.Linear(
                128,
                num_classes
            )
        )

    def forward(self, x):

        x = self.features(x)

        x = self.classifier(x)

        return x


# -----------------------------
# Evaluation
# -----------------------------

def evaluate(
    model,
    dataloader,
    criterion
):

    model.eval()

    total_loss = 0

    all_predictions = []
    all_labels = []

    with torch.no_grad():

        for images, labels in dataloader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            total_loss += loss.item()

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

    avg_loss = (
        total_loss / len(dataloader)
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

    return avg_loss, accuracy, macro_f1


# -----------------------------
# Training
# -----------------------------

def train():
    
    print("Using device:", DEVICE)
    
    train_loader, val_loader, _ = (
        create_dataloaders(
            batch_size=BATCH_SIZE,
            num_workers=2
        )
    )

    model = BaselineCNN().to(DEVICE)

    class_weights = (
        get_class_weights()
        .to(DEVICE)
    )

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    best_val_f1 = 0.0

    Path("models").mkdir(
        exist_ok=True
    )

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
                predictions.detach()
                .cpu()
                .numpy()
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

        val_loss, val_accuracy, val_f1 = (
            evaluate(
                model,
                val_loader,
                criterion
            )
        )

        print(
            f"\nEpoch [{epoch + 1}/{EPOCHS}]"
        )

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

        # Save best model based on validation macro F1
        if val_f1 > best_val_f1:

            best_val_f1 = val_f1

            torch.save(
                model.state_dict(),
                "models/baseline_cnn_best.pth"
            )

            print(
                "✓ Best model saved"
            )

    print(
        f"\nBest Validation Macro F1: "
        f"{best_val_f1:.4f}"
    )


if __name__ == "__main__":
    train()