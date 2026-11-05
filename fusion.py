import torch
import torch.nn as nn
import timm
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from pathlib import Path

# ── Paths ──
BASE = Path(r"D:\Web Dev Project\intermediate project\Skin disease detector")
EFFNET_PATH = BASE / "best_baseline.pth"      
SWIN_PATH   = BASE / "SwimTiny.pth"            
TEST_DIR    = BASE / "DermNet" / "test"
TRAIN_DIR   = BASE / "DermNet" / "train"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ── Get class count ──
train_ds = datasets.ImageFolder(TRAIN_DIR)
NUM_CLASSES = len(train_ds.classes)
print(f"Classes: {NUM_CLASSES}")

# ── Transforms (Swin uses 224, EfficientNet uses 224) ──
tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

test_ds = datasets.ImageFolder(TEST_DIR, transform=tf)
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)

# ── Load EfficientNet-B3 ──
effnet = timm.create_model("efficientnet_b3", pretrained=False, num_classes=NUM_CLASSES)
effnet.load_state_dict(torch.load(EFFNET_PATH, map_location=device))
effnet = effnet.to(device).eval()
print("✓ EfficientNet-B3 loaded")

# ── Load Swin Tiny ──
swin = timm.create_model("swin_tiny_patch4_window7_224", pretrained=False, num_classes=NUM_CLASSES)
swin.load_state_dict(torch.load(SWIN_PATH, map_location=device))
swin = swin.to(device).eval()
print("✓ Swin Tiny loaded")

# ── Fusion Prediction ──
def predict_fusion(imgs):
    with torch.no_grad():
        p_eff = torch.softmax(effnet(imgs), dim=1)
        p_swin = torch.softmax(swin(imgs), dim=1)
        # Weighted average — give Swin more weight since it's stronger
        fused = 0.40 * p_eff + 0.60 * p_swin
    return fused

# ── Evaluate Fusion ──
preds, truths = [], []
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs = imgs.to(device)
        fused = predict_fusion(imgs)
        preds.extend(fused.argmax(1).cpu().numpy())
        truths.extend(labels.numpy())

acc = accuracy_score(truths, preds) * 100
print(f"\n✅ Fusion Accuracy: {acc:.2f}%")