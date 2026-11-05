import torch
import timm
from torchvision import transforms, datasets
from PIL import Image
from pathlib import Path
import sys

# ── Paths ──
BASE = Path(r"D:\Web Dev Project\intermediate project\Skin disease detector")
MODEL_PATH = BASE / "SwimTiny.pth"
TRAIN_DIR = BASE / "DermNet" / "train"
TEST_DIR  = BASE / "DermNet" / "test"

# ── Device ──
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ── Load class names ──
train_ds = datasets.ImageFolder(TRAIN_DIR)
CLASSES = train_ds.classes
NUM_CLASSES = len(CLASSES)
print(f"Classes ({NUM_CLASSES}): {CLASSES}\n")

# ── Load model ──
model = timm.create_model("swin_tiny_patch4_window7_224", pretrained=False, num_classes=NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()
print("Model loaded successfully\n")

# ── Preprocessing (same as validation) ── 
tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# ── Prediction function ──
def predict(image_path):
    img = Image.open(image_path).convert("RGB")
    tensor = tf(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)[0]

    top3_probs, top3_idx = torch.topk(probs, 3)

    print(f"Image: {Path(image_path).name}")
    print(f"True label (if known): {Path(image_path).parent.name}")
    print("=" * 50)
    for i in range(3):
        bar = "█" * int(top3_probs[i] * 30)
        print(f"  {i+1}. {CLASSES[top3_idx[i]]:<20} {top3_probs[i]*100:6.2f}%  {bar}")
    print()


predict(r"D:\Web Dev Project\intermediate project\Skin disease detector\image for testing\image2.jpg")
