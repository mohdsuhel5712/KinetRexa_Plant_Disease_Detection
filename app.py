import os

from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

import torch
import torch.nn as nn

from PIL import Image
from torchvision import transforms


# =====================================================
# FLASK SETTINGS
# =====================================================

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
MODEL_PATH = "models/plant_disease_model.pth"

ALLOWED_EXTENSIONS = {
    "png",
    "jpg",
    "jpeg",
    "webp"
}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Create upload directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# =====================================================
# DEVICE
# =====================================================

# Render normally runs this application on CPU.
device = torch.device("cpu")

print("\nUsing device:", device)


# =====================================================
# CNN MODEL
# Must be EXACTLY the same architecture as train.py
# =====================================================

class PlantDiseaseCNN(nn.Module):

    def __init__(self, number_of_classes):

        super().__init__()

        self.cnn = nn.Sequential(

            # First CNN layer
            nn.Conv2d(
                3,
                32,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(2),


            # Second CNN layer
            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(2),


            # Third CNN layer
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

        image = self.cnn(image)

        output = self.classifier(image)

        return output


# =====================================================
# LOAD TRAINED MODEL
# =====================================================

print("\nLoading trained model...")

try:

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device
    )

    classes = checkpoint["classes"]

    image_size = checkpoint["image_size"]

    model = PlantDiseaseCNN(
        number_of_classes=len(classes)
    )

    model.load_state_dict(
        checkpoint["model_state"]
    )

    model.to(device)

    model.eval()

    print("Model loaded successfully!")

    print(
        "Disease classes:",
        classes
    )

except Exception as error:

    print(
        "ERROR: Could not load model:",
        error
    )

    raise error


# =====================================================
# IMAGE TRANSFORM
# =====================================================

transform = transforms.Compose([

    transforms.Resize(
        (
            image_size,
            image_size
        )
    ),

    transforms.ToTensor(),

    transforms.Normalize(

        mean=[
            0.5,
            0.5,
            0.5
        ],

        std=[
            0.5,
            0.5,
            0.5
        ]
    )
])


# =====================================================
# CHECK FILE EXTENSION
# =====================================================

def allowed_file(filename):

    return (
        "." in filename
        and
        filename.rsplit(".", 1)[1].lower()
        in ALLOWED_EXTENSIONS
    )


# =====================================================
# PREDICTION FUNCTION
# =====================================================

def predict_disease(image_path):

    image = Image.open(
        image_path
    ).convert("RGB")


    image_tensor = transform(
        image
    )


    image_tensor = image_tensor.unsqueeze(0)


    image_tensor = image_tensor.to(
        device
    )


    with torch.no_grad():

        output = model(
            image_tensor
        )


        probabilities = torch.softmax(
            output,
            dim=1
        )


        confidence, prediction = torch.max(
            probabilities,
            dim=1
        )


    predicted_disease = classes[
        prediction.item()
    ]


    confidence_percentage = (
        confidence.item() * 100
    )


    return (
        predicted_disease,
        round(
            confidence_percentage,
            2
        )
    )


# =====================================================
# HOME PAGE
# =====================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# =====================================================
# PREDICT DISEASE
# =====================================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    # Check if image exists
    if "leaf_image" not in request.files:

        return render_template(
            "index.html",
            error="Please select a leaf image."
        )


    file = request.files[
        "leaf_image"
    ]


    # Check empty filename
    if file.filename == "":

        return render_template(
            "index.html",
            error="Please select a leaf image."
        )


    # Check extension
    if not allowed_file(
        file.filename
    ):

        return render_template(
            "index.html",
            error=(
                "Invalid file. "
                "Upload PNG, JPG, JPEG, "
                "or WEBP image."
            )
        )


    # Secure filename
    filename = secure_filename(
        file.filename
    )


    file_path = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename
    )


    # Save image
    file.save(
        file_path
    )


    try:

        predicted_disease, confidence = (
            predict_disease(
                file_path
            )
        )


        image_url = (
            "uploads/" + filename
        )


        return render_template(
            "index.html",
            prediction=predicted_disease,
            confidence=confidence,
            image_url=image_url
        )


    except Exception as error:

        print(
            "Prediction error:",
            error
        )


        return render_template(
            "index.html",
            error=(
                "Unable to predict "
                "the uploaded image."
            )
        )


# =====================================================
# RUN FLASK APPLICATION
# =====================================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=int(
            os.environ.get(
                "PORT",
                5000
            )
        ),
        debug=False
    )