import torch
import torch.nn as nn
import pandas as pd

from torch.utils.data import DataLoader, Subset
from torchvision.models import resnet18, ResNet18_Weights

from image_preprocess import BananaDataset, transform


# ============================================================
# 1. DEVICE
# ============================================================

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("Using device:", device)


# ============================================================
# 2. LOAD DATASET
# ============================================================

dataset = BananaDataset(
    csv_file="banana_labels.csv",
    root_dir="BananaImages",
    transform=transform
)


# ============================================================
# 3. TEST BANANAS
# ============================================================

test_bananas = [
    "Banana010",
    "Banana011"
]

test_indices = dataset.data[
    dataset.data["banana_id"].isin(test_bananas)
].index.tolist()

test_dataset = Subset(
    dataset,
    test_indices
)

test_loader = DataLoader(
    test_dataset,
    batch_size=8,
    shuffle=False
)

print(
    "\nTest images:",
    len(test_dataset)
)


# ============================================================
# 4. CREATE RESNET18 MODEL
# ============================================================

weights = ResNet18_Weights.DEFAULT

model = resnet18(
    weights=weights
)

num_features = model.fc.in_features

model.fc = nn.Linear(
    num_features,
    1
)

model.load_state_dict(
    torch.load(
        "banana_resnet18_best.pth",
        map_location=device
    )
)

model = model.to(device)

model.eval()

print("\nBest model loaded successfully.")


# ============================================================
# 5. TEST MODEL
# ============================================================

results = []

total_absolute_error = 0.0
total_samples = 0


with torch.no_grad():

    for subset_position in range(
        len(test_dataset)
    ):

        image, target = test_dataset[
            subset_position
        ]

        original_index = test_indices[
            subset_position
        ]

        row = dataset.data.iloc[
            original_index
        ]

        image = (
            image
            .unsqueeze(0)
            .to(device)
        )

        prediction = model(
            image
        ).item()


        # Days remaining cannot be negative
        prediction = max(
            0.0,
            prediction
        )


        actual = float(
            target.item()
        )

        error = abs(
            prediction - actual
        )


        total_absolute_error += error

        total_samples += 1


        results.append({

            "banana_id":
                row["banana_id"],

            "day_number":
                int(row["day_number"]),

            "actual_days_left":
                actual,

            "predicted_days_left":
                round(prediction, 2),

            "absolute_error":
                round(error, 2),

            "image_id":
                row["image_id"]

        })


# ============================================================
# 6. TEST MAE
# ============================================================

test_mae = (
    total_absolute_error /
    total_samples
)


print("\n==============================")
print("TEST RESULTS")
print("==============================")

print(
    f"Test MAE: "
    f"{test_mae:.2f} days"
)


# ============================================================
# 7. RESULTS TABLE
# ============================================================

results_df = pd.DataFrame(
    results
)

print(
    "\nPredictions:\n"
)

print(
    results_df[
        [
            "banana_id",
            "day_number",
            "actual_days_left",
            "predicted_days_left",
            "absolute_error"
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# 8. SAVE RESULTS
# ============================================================

results_df.to_csv(
    "banana_test_predictions.csv",
    index=False
)

print(
    "\nResults saved as "
    "banana_test_predictions.csv"
)