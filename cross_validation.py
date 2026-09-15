import torch
import torch.nn as nn
import torch.optim as optim

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

print("\nUsing device:", device)


# ============================================================
# 2. LOAD CLEAN DATASET
# ============================================================

dataset = BananaDataset(
    csv_file="banana_labels.csv",
    root_dir="BananaImages",
    transform=transform
)

print("\nTotal usable images:", len(dataset))


# ============================================================
# 3. BANANAS USED FOR CROSS-VALIDATION
# ============================================================

# Keep Banana010 and Banana011 untouched as final test bananas.
# Banana009 is excluded for now because its labels are unusual.

cv_bananas = [
    "Banana001",
    "Banana002",
    "Banana003",
    "Banana004",
    "Banana005",
    "Banana006",
    "Banana007",
    "Banana008"
]


# ============================================================
# 4. SETTINGS
# ============================================================

num_epochs = 10
batch_size = 8

fold_results = []


# ============================================================
# 5. LEAVE-ONE-BANANA-OUT CROSS-VALIDATION
# ============================================================

for fold_number, val_banana in enumerate(cv_bananas, start=1):

    print("\n")
    print("=" * 60)
    print(
        f"FOLD {fold_number}/{len(cv_bananas)} "
        f"- Validation Banana: {val_banana}"
    )
    print("=" * 60)


    # --------------------------------------------------------
    # Training bananas = every banana except validation banana
    # --------------------------------------------------------

    train_bananas = [
        banana
        for banana in cv_bananas
        if banana != val_banana
    ]


    # --------------------------------------------------------
    # Get indices
    # --------------------------------------------------------

    train_indices = dataset.data[
        dataset.data["banana_id"].isin(train_bananas)
    ].index.tolist()

    val_indices = dataset.data[
        dataset.data["banana_id"] == val_banana
    ].index.tolist()


    train_dataset = Subset(
        dataset,
        train_indices
    )

    val_dataset = Subset(
        dataset,
        val_indices
    )


    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )


    print(
        "Training images:",
        len(train_dataset)
    )

    print(
        "Validation images:",
        len(val_dataset)
    )


    # ========================================================
    # 6. CREATE NEW MODEL FOR THIS FOLD
    # ========================================================

    weights = ResNet18_Weights.DEFAULT

    model = resnet18(
        weights=weights
    )


    # Freeze everything first
    for param in model.parameters():
        param.requires_grad = False


    # Unfreeze layer4
    for param in model.layer4.parameters():
        param.requires_grad = True


    # Regression output
    num_features = model.fc.in_features

    model.fc = nn.Linear(
        num_features,
        1
    )

    model = model.to(device)


    # ========================================================
    # 7. LOSS + OPTIMIZER
    # ========================================================

    criterion = nn.MSELoss()

    optimizer = optim.Adam(
        [
            {
                "params":
                model.layer4.parameters()
            },
            {
                "params":
                model.fc.parameters()
            }
        ],
        lr=0.0001
    )


    best_val_mae = float("inf")


    # ========================================================
    # 8. TRAIN THIS FOLD
    # ========================================================

    for epoch in range(num_epochs):

        model.train()

        train_absolute_error = 0.0
        train_samples = 0


        for images, targets in train_loader:

            images = images.to(device)

            targets = (
                targets
                .to(device)
                .unsqueeze(1)
            )


            optimizer.zero_grad()

            predictions = model(images)

            loss = criterion(
                predictions,
                targets
            )

            loss.backward()

            optimizer.step()


            train_absolute_error += (
                torch.abs(
                    predictions - targets
                )
                .sum()
                .item()
            )

            train_samples += images.size(0)


        train_mae = (
            train_absolute_error /
            train_samples
        )


        # ====================================================
        # VALIDATION
        # ====================================================

        model.eval()

        val_absolute_error = 0.0
        val_samples = 0


        with torch.no_grad():

            for images, targets in val_loader:

                images = images.to(device)

                targets = (
                    targets
                    .to(device)
                    .unsqueeze(1)
                )


                predictions = model(images)


                val_absolute_error += (
                    torch.abs(
                        predictions - targets
                    )
                    .sum()
                    .item()
                )

                val_samples += images.size(0)


        val_mae = (
            val_absolute_error /
            val_samples
        )


        if val_mae < best_val_mae:
            best_val_mae = val_mae


        print(
            f"Epoch [{epoch + 1}/{num_epochs}] "
            f"Train MAE: {train_mae:.2f} days | "
            f"Val MAE: {val_mae:.2f} days"
        )


    # ========================================================
    # 9. SAVE FOLD RESULT
    # ========================================================

    fold_results.append(
        {
            "banana":
                val_banana,

            "best_mae":
                best_val_mae
        }
    )


    print(
        f"\nBest MAE for {val_banana}: "
        f"{best_val_mae:.2f} days"
    )


# ============================================================
# 10. FINAL CROSS-VALIDATION RESULTS
# ============================================================

print("\n")
print("=" * 60)
print("CROSS-VALIDATION RESULTS")
print("=" * 60)


total_mae = 0.0


for result in fold_results:

    print(
        f"{result['banana']}: "
        f"{result['best_mae']:.2f} days"
    )

    total_mae += result["best_mae"]


average_mae = (
    total_mae /
    len(fold_results)
)


print("-" * 60)

print(
    f"Average Cross-Validation MAE: "
    f"{average_mae:.2f} days"
)