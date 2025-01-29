import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import random
import matplotlib.pyplot as plt

# Define a transform to normalize the data
transform = transforms.Compose([
    transforms.ToTensor(),  # convert image to PyTorch tensor
    transforms.Normalize((0.1307,), (0.3081,))  # normalize with mean and std for MNIST
])

# Download and create training and test datasets
train_dataset = datasets.MNIST(root='/Users/lanl2tz/Documents/anaconda/example/cnn/dataset', 
                               train=True, 
                               download=False, 
                               transform=transform)
test_dataset = datasets.MNIST(root='/Users/lanl2tz/Documents/anaconda/example/cnn/dataset', 
                              train=False, 
                              download=False, 
                              transform=transform)

# Data loaders
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

# Define the model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(32 * 7 * 7, 128)  # after two poolings, the size is 7x7
        self.fc2 = nn.Linear(128, 10)         # 10 classes for digits 0-9

    def forward(self, x):
        # First convolution + ReLU + pool
        x = self.pool(nn.functional.relu(self.conv1(x)))
        # Second convolution + ReLU + pool
        x = self.pool(nn.functional.relu(self.conv2(x)))
        
        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)
        
        # Fully connected layers with ReLU in-be tween
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize the model
model = SimpleCNN()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

num_epochs = 5

for epoch in range(num_epochs):
    model.train()  # set the model in training mode
    running_loss = 0.0
    
    for images, labels in train_loader:
        # Zero out the gradients from the previous iteration
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        
        # Update the parameters
        optimizer.step()
        
        running_loss += loss.item()

    # Calculate the average loss for this epoch
    epoch_loss = running_loss / len(train_loader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')

# Test the model
model.eval()  # set the model to evaluation mode
correct = 0
total = 0

with torch.no_grad():  # disable gradient calculation
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f'Test Accuracy: {accuracy:.2f}%')

# ------------------------------
# Pick Random Images & Infer
# ------------------------------
num_samples_to_show = 5
indices = random.sample(range(len(test_dataset)), num_samples_to_show)

plt.figure(figsize=(12, 3))

for i, idx in enumerate(indices, start=1):
    # Get an image-label pair from the dataset
    image, label = test_dataset[idx]  # image is [1, 28, 28]
    
    # Model inference
    with torch.no_grad():
        # Add batch dimension: shape becomes [1, 1, 28, 28]
        output = model(image.unsqueeze(0))
        # Predicted class
        _, predicted_class = torch.max(output, 1)

    # Convert single-channel tensor to displayable image
    img_display = image.squeeze(0).numpy()  # shape [28, 28]
    
    # Plot
    plt.subplot(1, num_samples_to_show, i)
    plt.imshow(img_display, cmap="gray")
    plt.title(f"Pred: {predicted_class.item()} / True: {label}")
    plt.axis("off")

plt.tight_layout()
plt.show()