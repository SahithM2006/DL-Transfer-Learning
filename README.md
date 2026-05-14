# DL- Developing a Neural Network Classification Model using Transfer Learning

## AIM
To develop an image classification model using transfer learning with VGG19 architecture for the given dataset.

## Problem Statement and Dataset
Developing a Neural Network Classification Model using Transfer Learning

Training deep neural networks from scratch requires large datasets, high computational power, and significant training time. In many practical scenarios, such resources are limited.

The goal of this project is to develop an image classification model using Transfer Learning, where a pre-trained neural network is reused and fine-tuned to classify new data efficiently.


## Neural Network Model
A Neural Network Model is a type of machine learning model inspired by the structure and functioning of the human brain. It is used to learn patterns from data and make predictions or decisions.
<img width="1536" height="1024" alt="image" src="https://github.com/user-attachments/assets/9eafed7e-8922-420c-a06d-9751f285ebd3" />

## DESIGN STEPS
STEP 1:
Import required libraries and define image transforms.

STEP 2:
Load training and testing datasets using ImageFolder.

STEP 3:
Visualize sample images from the dataset.

STEP 4:
Load pre-trained VGG19, modify the final layer for binary classification, and freeze feature extractor layers.

STEP 5:
Define loss function (BCEWithLogitsLoss) and optimizer (Adam). Train the model and plot the loss curve.

STEP 6:
Evaluate the model with test accuracy, confusion matrix, classification report, and visualize predictions.


## PROGRAM

### Name: SAHITH M 

### Register Number: 212224230236

```# Load Pretrained Model and Modify for Transfer Learning

model=models.vgg19(weights=VGG19_Weights.DEFAULT)

# Modify the final fully connected layer to match the dataset classes

model.classifier[-1]=nn.Linear(model.classifier[-1].in_features,1)

# Include the Loss function and optimizer
criterion =nn.BCEWithLogitsLoss()
optimizer =optim.Adam(model.parameters(),lr=0.001)


# Train the model
def train_model(model, train_loader,test_loader,num_epochs=100):
    train_losses = []
    val_losses = []
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels.unsqueeze(1).float())

            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_losses.append(running_loss / len(train_loader))

        # compute validation loss
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels.unsqueeze(1).float())
                val_loss += loss.item()
        val_losses.append(val_loss / len(test_loader))
        model.train()

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Validation Loss: {val_losses[-1]:.4f}')

    # Plot training and validation loss
    print("Name: PRAVEEN RAJ R ")
    print("Register Number: 212224230207")
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss', marker='o')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss', marker='s')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()

```

### OUTPUT

## Training Loss, Validation Loss Vs Iteration Plot

<img width="1032" height="717" alt="image" src="https://github.com/user-attachments/assets/66118133-b5ce-4d0c-9069-74940e434380" />

## Confusion Matrix

<img width="720" height="682" alt="image" src="https://github.com/user-attachments/assets/3eecdc42-b9ed-4068-a1f3-acc6ce3f3579" />


## Classification Report
<img width="1045" height="291" alt="image" src="https://github.com/user-attachments/assets/a20140b4-dddb-4c7f-8f92-eb2ba31dc1b3" />


### New Sample Data Prediction
<img width="565" height="890" alt="image" src="https://github.com/user-attachments/assets/da7ca83c-ce0b-4f54-ba8f-b0e26037a168" />


## RESULT
The image classification model using transfer learning with VGG19 architecture for the given dataset has been executed successfully.
