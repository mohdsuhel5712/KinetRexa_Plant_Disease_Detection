
# =====================================================
# train.py
# Plant Disease Detection - CNN Training
# =====================================================
# HERE I CAN DEFINE HOW CANNTRAIN A CNN MODEL 

import os
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# =====================================================
# SETTINGS
# =====================================================

CSV_FILE = "dataset/data.csv"
MODEL_PATH = "models/plant_disease_model.pth"

IMAGE_SIZE = 128
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 0.001


# =====================================================
# CREATE CUSTOM DATASET
# =====================================================
# i am processing data in csv 
class PlantDataset(Dataset):

    def __init__(self, csv_file, transform=None):

        self.data = pd.read_csv(csv_file)
        self.transform = transform

        # Get all unique disease names
        self.classes = sorted(
            self.data["label"].unique()
        ) # means , category of the disease

        # Convert disease name into number
        self.class_to_index = {
            class_name: index
            for index, class_name
            in enumerate(self.classes)
        }


    def __len__(self):
        return len(self.data)


    def __getitem__(self, index):
        # Get image path from CSV
        image_path = self.data.iloc[index]["image_path"]

        # CSV path is relative to dataset folder
        full_image_path = os.path.join(
            "dataset",
            image_path
        )

        # Open image
        image = Image.open(
            full_image_path
        ).convert("RGB")

        # Get disease label
        label_name = self.data.iloc[index]["label"]

        # Convert label into number
        label = self.class_to_index[
            label_name
        ]

        # Apply image processing
        if self.transform:
            image = self.transform(
                image
            )

        return image, label

''' custom dataset has been been completed  '''
# =====================================================
# IMAGE PROCESSING
# =====================================================
'''agar dekha jaye yaha per abho toh , kewal variable define hua hai'''
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])
# image processing ke liye variable defining


# =====================================================
# LOAD DATASET
# =====================================================
# making object of the class PlantDataset()
dataset = PlantDataset(CSV_FILE, transform)

# object of the DataLoader()
data_loader = DataLoader(dataset,batch_size=BATCH_SIZE,shuffle=True)


# =====================================================
# CNN MODEL
# =====================================================
# here actual model has been created 
'''after processing , loading data '''

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

        image = self.cnn(
            image
        )

        output = self.classifier(
            image
        )

        return output


# =====================================================
# DEVICE
# =====================================================
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print(
    "\nUsing device:",
    device
)


# =====================================================
# CREATE MODEL
# =====================================================

number_of_classes = len(dataset.classes)
# making of objct of the , CCN MODEL 
model = PlantDiseaseCNN(number_of_classes)

model = model.to(device)


# =====================================================
# LOSS AND OPTIMIZER
# =====================================================
# making object of the los function 
loss_function = nn.CrossEntropyLoss()

# also making variable of the optim
optimizer = optim.Adam(model.parameters(),lr=LEARNING_RATE)


# =====================================================
# TRAIN MODEL
# =====================================================
print(
    "\nTraining started..."
)


for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    correct_predictions = 0
    total_images = 0

    for images, labels in data_loader:

        images = images.to(device)
        labels = labels.to(device)


        # Clear old gradients
        optimizer.zero_grad()

        # Model prediction
        outputs = model(images)

        # Calculate loss
        loss = loss_function(outputs,labels)
        # Backpropagation
        loss.backward()
        # Update model
        optimizer.step()
        total_loss += (loss.item())
        # Calculate accuracy
        predicted_labels = torch.argmax(outputs,dim=1)

        correct_predictions += (predicted_labels == labels).sum().item()


        total_images += (labels.size(0))


    accuracy = (correct_predictions/ total_images)* 100


    average_loss = (total_loss/ len(data_loader))


    print(f"Epoch "f"{epoch + 1}/{EPOCHS}"f" | Loss: "f"{average_loss:.4f}"f" | Accuracy: "f"{accuracy:.2f}%"
    )


# =====================================================
# SAVE MODEL
# =====================================================

os.makedirs(
    "models",
    exist_ok=True
)


torch.save(
    {
        "model_state": model.state_dict(),
        "classes": dataset.classes,
        "image_size": IMAGE_SIZE
    },
    MODEL_PATH
)


print(

    "\nModel trained successfully!"

)

print(

    "Model saved at:",

    MODEL_PATH

)

print(

    "Disease classes:",

    dataset.classes

)
 