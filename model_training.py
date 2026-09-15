import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, Subset
from torchvision.models import resnet18, ResNet18_Weights

from image_preprocess import BananaDataset, transform


# ============================================================
# 1. CREATE DATASET
# ============================================================

dataset = BananaDataset(
    csv_file="banana_labels.csv",
    root_dir="BananaImages",
    transform=transform
)

print("\nTotal valid images:", len(dataset))


# ============================================================
# 2. SPLIT BY BANANA ID
# ============================================================

train_bananas = [
    "Banana001",
    "Banana002",
    "Banana003",
    "Banana004",
    "Banana005",
    "Banana006",
    "Banana007"
]

val_bananas = [
    "Banana008"
]

test_bananas = [
    "Banana010",
    "Banana011"
]


train_indices = dataset.data[
    dataset.data["banana_id"].isin(train_bananas)
].index.tolist()

val_indices = dataset.data[
    dataset.data["banana_id"].isin(val_bananas)
].index.tolist()

test_indices = dataset.data[
    dataset.data["banana_id"].isin(test_bananas)
].index.tolist()


train_dataset = Subset(
    dataset,
    train_indices
)

val_dataset = Subset(
    dataset,
    val_indices
)

test_dataset = Subset(
    dataset,
    test_indices
)


# ============================================================
# 3. DATALOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=8,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=8,
    shuffle=False
)


print("\nDataset Split")
print("-----------------------------")
print("Training images:", len(train_dataset))
print("Validation images:", len(val_dataset))
print("Test images:", len(test_dataset))


# ============================================================
# 4. DEVICE
# ============================================================

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("\nUsing device:", device)


# ============================================================
# 5. LOAD PRETRAINED RESNET18
# ============================================================

weights = ResNet18_Weights.DEFAULT

model = resnet18(
    weights=weights
)


for param in model.parameters():
    param.requires_grad = False

for param in model.layer4.parameters():
    param.requires_grad = True

# ============================================================
# 6. CHANGE FINAL LAYER TO REGRESSION
# ============================================================

num_features = model.fc.in_features

model.fc = nn.Linear(
    num_features,
    1
)

model = model.to(device)

print("\nResNet18 regression model ready")


# ============================================================
# 7. LOSS + OPTIMIZER
# ============================================================

criterion = nn.MSELoss()

optimizer = optim.Adam(
    [
        {"params": model.layer4.parameters()},
        {"params": model.fc.parameters()}
    ],
    lr=0.0001
)

num_epochs = 10

best_val_mae = float("inf")

for epoch in range(num_epochs):

    # =========================
    # TRAINING
    # =========================

    model.train()

    train_loss = 0.0
    train_absolute_error = 0.0
    train_samples = 0

    for images, targets in train_loader:

        images = images.to(device)
        targets = targets.to(device).unsqueeze(1)

        optimizer.zero_grad()

        predictions = model(images)

        loss = criterion(predictions, targets)

        loss.backward()

        optimizer.step()

        batch_size = images.size(0)

        train_loss += loss.item() * batch_size

        train_absolute_error += (
            torch.abs(predictions - targets)
            .sum()
            .item()
        )

        train_samples += batch_size


    average_train_loss = train_loss / train_samples

    average_train_mae = (
        train_absolute_error / train_samples
    )


    # =========================
    # VALIDATION
    # =========================

    model.eval()

    val_loss = 0.0
    val_absolute_error = 0.0
    val_samples = 0

    with torch.no_grad():

        for images, targets in val_loader:

            images = images.to(device)
            targets = targets.to(device).unsqueeze(1)

            predictions = model(images)

            loss = criterion(
                predictions,
                targets
            )

            batch_size = images.size(0)

            val_loss += (
                loss.item() * batch_size
            )

            val_absolute_error += (
                torch.abs(predictions - targets)
                .sum()
                .item()
            )

            val_samples += batch_size


    average_val_loss = val_loss / val_samples

    average_val_mae = (
        val_absolute_error / val_samples
    )


    print(
        f"Epoch [{epoch + 1}/{num_epochs}] "
        f"Train Loss: {average_train_loss:.4f} "
        f"Train MAE: {average_train_mae:.2f} days "
        f"Val Loss: {average_val_loss:.4f} "
        f"Val MAE: {average_val_mae:.2f} days"
    )


    # =========================
    # SAVE BEST MODEL
    # =========================

    if average_val_mae < best_val_mae:

        best_val_mae = average_val_mae

        torch.save(
            model.state_dict(),
            "banana_resnet18_best.pth"
        )

        print(
            f"  -> Best model saved "
            f"(Val MAE: {best_val_mae:.2f} days)"
        )


print(
    "\nTraining finished."
)

print(
    f"Best Validation MAE: "
    f"{best_val_mae:.2f} days"
)