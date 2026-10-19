import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from data import CLASS_NAMES


CM_PATH = "outputs/predictions/confusion_matrix.csv"
OUTPUT_PATH = "outputs/figures/confusion_matrix.png"


def main():

    cm = np.loadtxt(
        CM_PATH,
        delimiter=",",
        dtype=int
    )

    Path("outputs/figures").mkdir(
        parents=True,
        exist_ok=True
    )

    plt.figure(figsize=(12, 10))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES
    )

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Waste Classification - Confusion Matrix")

    plt.tight_layout()

    plt.savefig(
        OUTPUT_PATH,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"✓ Confusion matrix saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()