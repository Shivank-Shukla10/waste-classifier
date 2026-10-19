import mlflow
import mlflow.pytorch

from pathlib import Path


# -----------------------------
# MLflow configuration
# -----------------------------

mlflow.set_tracking_uri(
    "sqlite:///mlflow.db"
)

mlflow.set_experiment(
    "Waste-Classification"
)


# -----------------------------
# Actual experiment results
# -----------------------------

VAL_ACCURACY = 0.9244
VAL_MACRO_F1 = 0.8971

TEST_ACCURACY = 0.9218
TEST_MACRO_F1 = 0.8934


def main():

    model_path = Path(
        "models/mobilenetv2_finetuned_best.pth"
    )

    with mlflow.start_run(
        run_name="MobileNetV2_FineTuned_Final"
    ):

        # -----------------------------
        # Parameters
        # -----------------------------

        mlflow.log_params({

            "model": "MobileNetV2",

            "transfer_learning": True,

            "fine_tuning": True,

            "image_size": 160,

            "batch_size": 64,

            "epochs": 4,

            "learning_rate": 1e-5,

            "optimizer": "Adam",

            "loss": "Weighted Cross Entropy",

            "num_classes": 12,

            "train_images": 10860,

            "validation_images": 2327,

            "test_images": 2328
        })


        # -----------------------------
        # Metrics
        # -----------------------------

        mlflow.log_metrics({

            "val_accuracy": VAL_ACCURACY,

            "val_macro_f1": VAL_MACRO_F1,

            "test_accuracy": TEST_ACCURACY,

            "test_macro_f1": TEST_MACRO_F1
        })


        # -----------------------------
        # Artifacts
        # -----------------------------

        artifacts = [

            "outputs/figures/confusion_matrix.png",

            "outputs/predictions/classification_report.txt",

            "outputs/predictions/confusion_matrix.csv"
        ]

        for artifact in artifacts:

            artifact_path = Path(artifact)

            if artifact_path.exists():

                mlflow.log_artifact(
                    str(artifact_path)
                )


        # Grad-CAM artifacts

        gradcam_dir = Path(
            "outputs/figures/gradcam"
        )

        if gradcam_dir.exists():

            for image in gradcam_dir.glob("*.jpg"):

                mlflow.log_artifact(
                    str(image),
                    artifact_path="gradcam"
                )


        # Model checkpoint

        if model_path.exists():

            mlflow.log_artifact(
                str(model_path),
                artifact_path="model"
            )


        print("\n✓ MLflow tracking completed")

        print(
            f"Run ID: {mlflow.active_run().info.run_id}"
        )


if __name__ == "__main__":
    main()