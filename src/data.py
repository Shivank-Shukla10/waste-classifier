import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.utils.class_weight import compute_class_weight
from pathlib import Path
from PIL import Image

DATA_DIR = Path("data/raw")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def inspect_dataset():
    class_counts = {}
    corrupted_images = []
    total_images = 0

    class_dirs = sorted(
        [folder for folder in DATA_DIR.iterdir() if folder.is_dir()]
    )

    print(f"Number of classes: {len(class_dirs)}")
    print("\nClasses:")

    for class_dir in class_dirs:
        images = [
            file for file in class_dir.iterdir()
            if file.suffix.lower() in IMAGE_EXTENSIONS
        ]

        class_counts[class_dir.name] = len(images)
        total_images += len(images)

        for image_path in images:
            try:
                with Image.open(image_path) as img:
                    img.verify()
            except Exception:
                corrupted_images.append(str(image_path))

    print("\nClass Distribution:")
    print("-" * 40)

    for class_name, count in class_counts.items():
        print(f"{class_name:15} : {count}")

    print("-" * 40)
    print(f"Total images      : {total_images}")
    print(f"Corrupted images  : {len(corrupted_images)}")

    if corrupted_images:
        print("\nCorrupted files:")
        for image in corrupted_images:
            print(image)

def inspect_image_dimensions():
    dimensions = {}

    class_dirs = sorted(
        [folder for folder in DATA_DIR.iterdir() if folder.is_dir()]
    )

    for class_dir in class_dirs:
        dimensions[class_dir.name] = []

        images = [
            file for file in class_dir.iterdir()
            if file.suffix.lower() in IMAGE_EXTENSIONS
        ]

        for image_path in images[:100]:
            try:
                with Image.open(image_path) as img:
                    dimensions[class_dir.name].append(img.size)
            except Exception:
                pass

    print("\nSample Image Dimensions:")
    print("-" * 50)

    for class_name, sizes in dimensions.items():
        unique_sizes = set(sizes)

        print(
            f"{class_name:15} : "
            f"{len(unique_sizes)} unique dimensions "
            f"| Examples: {list(unique_sizes)[:3]}"
        )


def create_stratified_split():

    records = []

    class_dirs = sorted(
        [folder for folder in DATA_DIR.iterdir() if folder.is_dir()]
    )

    for class_dir in class_dirs:

        images = [
            file for file in class_dir.iterdir()
            if file.suffix.lower() in IMAGE_EXTENSIONS
        ]

        for image_path in images:
            records.append({
                "image_path": str(image_path),
                "class": class_dir.name
            })

    df = pd.DataFrame(records)

    # First split: 70% train, 30% temporary
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        stratify=df["class"],
        random_state=42
    )

    # Second split: 15% validation, 15% test
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        stratify=temp_df["class"],
        random_state=42
    )

    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(
        output_dir / "train.csv",
        index=False
    )

    val_df.to_csv(
        output_dir / "val.csv",
        index=False
    )

    test_df.to_csv(
        output_dir / "test.csv",
        index=False
    )

    print("\nDataset Split")
    print("-" * 40)

    print(f"Total      : {len(df)}")
    print(f"Train      : {len(train_df)}")
    print(f"Validation : {len(val_df)}")
    print(f"Test       : {len(test_df)}")

    print("\nTrain distribution:")
    print(train_df["class"].value_counts())

    print("\nValidation distribution:")
    print(val_df["class"].value_counts())

    print("\nTest distribution:")
    print(test_df["class"].value_counts())


# 5.2 Add the class mapping

CLASS_NAMES = [
    "battery",
    "biological",
    "brown-glass",
    "cardboard",
    "clothes",
    "green-glass",
    "metal",
    "paper",
    "plastic",
    "shoes",
    "trash",
    "white-glass"
]

CLASS_TO_IDX = {
    class_name: idx
    for idx, class_name in enumerate(CLASS_NAMES)
}

IDX_TO_CLASS = {
    idx: class_name
    for class_name, idx in CLASS_TO_IDX.items()
}


# 5.3 Create the Dataset class

class WasteDataset(Dataset):

    def __init__(self, dataframe, transform=None):

        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):

        row = self.dataframe.iloc[index]

        image_path = row["image_path"]
        class_name = row["class"]

        image = Image.open(image_path).convert("RGB")

        label = CLASS_TO_IDX[class_name]

        if self.transform:
            image = self.transform(image)

        return image, label


# 5.4 Define transformations

IMAGE_SIZE = 160

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(
        IMAGE_SIZE,
        scale=(0.8, 1.0)
    ),
    transforms.RandomHorizontalFlip(
        p=0.5
    ),
    transforms.RandomRotation(
        degrees=15
    ),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


val_test_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# 5.5 Create DataLoaders

def create_dataloaders(
    batch_size=32,
    num_workers=0
):

    train_df = pd.read_csv(
        "data/processed/train.csv"
    )

    val_df = pd.read_csv(
        "data/processed/val.csv"
    )

    test_df = pd.read_csv(
        "data/processed/test.csv"
    )

    train_dataset = WasteDataset(
        train_df,
        transform=train_transform
    )

    val_dataset = WasteDataset(
        val_df,
        transform=val_test_transform
    )

    test_dataset = WasteDataset(
        test_df,
        transform=val_test_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, val_loader, test_loader


# 5.6 Calculate class weights

def get_class_weights():

    train_df = pd.read_csv(
        "data/processed/train.csv"
    )

    labels = train_df["class"].map(
        CLASS_TO_IDX
    ).values

    weights = compute_class_weight(
        class_weight="balanced",
        classes=torch.arange(len(CLASS_NAMES)).numpy(),
        y=labels
    )

    class_weights = torch.tensor(
        weights,
        dtype=torch.float32
    )

    return class_weights


if __name__ == "__main__":

    inspect_dataset()
    inspect_image_dimensions()
    create_stratified_split()

    train_loader, val_loader, test_loader = create_dataloaders()

    print("\nDataLoader Test")
    print("-" * 40)

    print("Train batches:", len(train_loader))
    print("Validation batches:", len(val_loader))
    print("Test batches:", len(test_loader))

    images, labels = next(iter(train_loader))

    print("Image batch shape:", images.shape)
    print("Label batch shape:", labels.shape)

    print("\nClass weights:")
    print(get_class_weights())