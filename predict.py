# =====================================================
# predict.py
# Plant Disease Prediction
# =====================================================
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL  import Image
# =====================================================
# SETTINGS
# =====================================================
MODEL_PATH = "models/plant_disease_model.pth"
# =====================================================
# CNN MODEL
# Must be same as train.py
# =====================================================
class PlantDiseaseCNN(nn.Module):
    def __init__(self, number_of_classes):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(
                3,
                32,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                128 * 16 * 16,
                256
            ),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(
                256,
                number_of_classes
            )
        )
    def forward(self, image):
        image = self.cnn(
            image
        )
        output = self.classifier(
            image
        )
        return output
# =====================================================
# LOAD SAVED MODEL
# =====================================================
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

checkpoint = torch.load(MODEL_PATH,map_location=device)

classes = checkpoint["classes"]

image_size = checkpoint["image_size"]

model = PlantDiseaseCNN(
    len(classes)
)

model.load_state_dict(checkpoint["model_state"])


model.to(
    device
)
model.eval()
# =====================================================
# IMAGE PROCESSING
# =====================================================
transform = transforms.Compose([
    transforms.Resize(
        (image_size, image_size)
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])
# =====================================================
# GET IMAGE PATH
# =====================================================
image_path = input(
    "\nEnter leaf image path: "
)
# =====================================================
# LOAD IMAGE
# =====================================================
try:
    image = Image.open(
        image_path
    ).convert(
        "RGB"
    )
except FileNotFoundError:
    print(
        "\nImage not found!"
    )
    print(
        "Check image path."
    )
    exit()
# =====================================================
# PREPARE IMAGE
# =====================================================
image = transform(image)

image = image.unsqueeze(0)

image = image.to(device)
# =====================================================
# PREDICTION
# =====================================================
with torch.no_grad():
    output = model(image)
    probabilities = torch.softmax(output,dim=1)
    confidence, prediction = torch.max(probabilities,dim=1)
# =====================================================
# DISPLAY RESULT
# =====================================================
predicted_disease = classes[
    prediction.item()
]
confidence_percentage = (
    confidence.item()
    * 100
)


image = Image.open(image_path)
plt.figure(figsize=(8, 8))
plt.imshow(image)
plt.axis("off")
plt.show()




print(
    "\n=============================="
)
print(
    "PLANT DISEASE RESULT"
)
print(
    "=============================="
)
print(
    "Prediction:",
    predicted_disease
)
print(
    f"Confidence: "
    f"{confidence_percentage:.2f}%"
)
print(
    "==============================\n"
)
