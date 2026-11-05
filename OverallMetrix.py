import torch
import timm
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from pathlib import Path
import numpy as np

# ── Paths ──
BASE = Path(r"D:\Web Dev Project\intermediate project\Skin disease detector")
MODEL_PATH = BASE / "SwimTiny.pth"
TRAIN_DIR = BASE / "DermNet" / "train"
TEST_DIR  = BASE / "DermNet" / "test"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")

# ── Load class names ──
train_ds = datasets.ImageFolder(TRAIN_DIR)
CLASSES = train_ds.classes
NUM_CLASSES = len(CLASSES)

# ── Model ──
model = timm.create_model("swin_tiny_patch4_window7_224", pretrained=False, num_classes=NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device).eval()
print("✓ Model loaded\n")

# ── Data ──
tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])
test_ds = datasets.ImageFolder(TEST_DIR, transform=tf)
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)

# ── Evaluate ──
preds, truths = [], []
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs = imgs.to(device)
        out = model(imgs).argmax(1).cpu().numpy()
        preds.extend(out)
        truths.extend(labels.numpy())

# ── Metrics ──
acc = accuracy_score(truths, preds) * 100
precision = precision_score(truths, preds, average='weighted', zero_division=0) * 100
recall = recall_score(truths, preds, average='weighted', zero_division=0) * 100
f1 = f1_score(truths, preds, average='weighted', zero_division=0) * 100

print("=" * 50)
print("OVERALL METRICS")
print("=" * 50)
print(f"Accuracy:  {acc:.2f}%")
print(f"Precision: {precision:.2f}%")
print(f"Recall:    {recall:.2f}%")
print(f"F1 Score:  {f1:.2f}%")
print()

# ── Per-class metrics ──
print("=" * 50)
print("PER-CLASS METRICS")
print("=" * 50)
print(classification_report(truths, preds, target_names=CLASSES, zero_division=0))

# ── Confusion matrix ──
cm = confusion_matrix(truths, preds)
print("=" * 50)
print("CONFUSION MATRIX (saved to confusion_matrix.npy)")
print("=" * 50)
np.save(BASE / "confusion_matrix.npy", cm)
print(f"Shape: {cm.shape}")
print(f"Saved to: {BASE / 'confusion_matrix.npy'}")