import os
import csv
import pandas as pd
import torch

from torch.utils.data import Dataset
from torchvision.models import ResNet18_Weights
from PIL import Image


class BananaDataset(Dataset):

    def __init__(self, csv_file, root_dir, transform=None):

        # ---------------------------------------------
        # READ CSV
        # ---------------------------------------------

        with open(
            csv_file,
            "r",
            newline="",
            encoding="utf-8-sig"
        ) as file:

            reader = csv.reader(file)

            header = next(reader)

            header = [
                col.strip()
                for col in header
                if col.strip() != ""
            ]

            rows = []

            for row in reader:

                # Remove trailing empty field
                while (
                    len(row) > len(header)
                    and row[-1].strip() == ""
                ):
                    row.pop()

                if len(row) == len(header):
                    rows.append(row)


        self.data = pd.DataFrame(
            rows,
            columns=header
        )


        # ---------------------------------------------
        # CONVERT NUMERIC COLUMNS
        # ---------------------------------------------

        self.data["day_number"] = pd.to_numeric(
            self.data["day_number"],
            errors="coerce"
        )

        self.data["estimated_days_left"] = pd.to_numeric(
            self.data["estimated_days_left"],
            errors="coerce"
        )


        self.data = self.data.dropna(
            subset=[
                "image_id",
                "banana_id",
                "day_number",
                "estimated_days_left"
            ]
        )


        # ---------------------------------------------
        # FIND VALID IMAGES
        # ---------------------------------------------

        valid_rows = []

        bad_images = []

        for index, row in self.data.iterrows():

            image_path = os.path.join(
                root_dir,
                row["image_id"]
            )

            try:

                with Image.open(image_path) as img:

                    # Actually decode image
                    img.load()

                valid_rows.append(index)

            except Exception:

                bad_images.append(
                    row["image_id"]
                )


        # Keep ONLY valid images
        self.data = (
            self.data
            .loc[valid_rows]
            .reset_index(drop=True)
        )


        self.root_dir = root_dir

        self.transform = transform


        # ---------------------------------------------
        # REPORT
        # ---------------------------------------------

        print("\nDataset check")
        print("--------------------------")

        print(
            "Valid images:",
            len(self.data)
        )

        print(
            "Bad images removed:",
            len(bad_images)
        )

        if bad_images:

            print("\nBad image files:")

            for path in bad_images:
                print(path)


    def __len__(self):

        return len(self.data)


    def __getitem__(self, idx):

        row = self.data.iloc[idx]

        image_path = os.path.join(
            self.root_dir,
            row["image_id"]
        )


        image = Image.open(
            image_path
        ).convert("RGB")


        if self.transform:

            image = self.transform(
                image
            )


        target = torch.tensor(
            float(
                row["estimated_days_left"]
            ),
            dtype=torch.float32
        )


        return image, target


# ==============================================
# RESNET18 PREPROCESSING
# ==============================================

weights = ResNet18_Weights.DEFAULT

transform = weights.transforms()


# ==============================================
# TEST
# ==============================================

if __name__ == "__main__":

    dataset = BananaDataset(
        csv_file="banana_labels.csv",
        root_dir="BananaImages",
        transform=transform
    )

    print(
        "\nFinal usable images:",
        len(dataset)
    )

    if len(dataset) > 0:

        image, target = dataset[0]

        print(
            "Image shape:",
            image.shape
        )

        print(
            "Target:",
            target
        )