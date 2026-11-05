import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import timm
from sklearn.metrics import accuracy_score
from pathlib import Path

BASE = Path(r"D:\Web Dev Project\intermediate project\Skin disease detector")
TRAIN_DIR = BASE / "DermNet" / "train"
TEST_DIR  = BASE / "DermNet" / "test"

print(f"Train dir: {TRAIN_DIR}")
print(f"Test dir:  {TEST_DIR}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

#Transforms
train_tf = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])
val_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

#Data
train_ds = datasets.ImageFolder(TRAIN_DIR, transform=train_tf)
test_ds  = datasets.ImageFolder(TEST_DIR,  transform=val_tf)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,  num_workers=0)
test_loader  = DataLoader(test_ds,  batch_size=32, shuffle=False, num_workers=0)

NUM_CLASSES = len(train_ds.classes)
print(f"Classes: {NUM_CLASSES}")
print(f"Train images: {len(train_ds)}")
print(f"Test images:  {len(test_ds)}")

#Model
model = timm.create_model("xception41", pretrained=True, num_classes=NUM_CLASSES)
model = model.to(device)

#Loss & Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

#Training
EPOCHS = 20
best_acc = 0

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(imgs), labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    scheduler.step()

    model.eval()
    preds, truths = [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            out = model(imgs).argmax(1).cpu().numpy()
            preds.extend(out)
            truths.extend(labels.numpy())

    acc = accuracy_score(truths, preds) * 100
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {running_loss/len(train_loader):.4f} | Test Acc: {acc:.2f}%")

    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), BASE / "xception.pth")
        print(f"  ✓ Saved new best model ({acc:.2f}%)")

print(f"\nTraining done. Best Accuracy: {best_acc:.2f}%")