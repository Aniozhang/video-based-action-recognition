
import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from dataset.dataset import VideoDataset
from models.model import CNN_LSTM
from glob import glob
import numpy as np
import tqdm

BATCH_SIZE = 8
EPOCHS = 15
LR = 0.0005
NUM_CLASSES = 5

def train(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss, correct = 0, 0
    for videos, labels in train_loader:
        videos, labels = videos.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(videos)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += (outputs.argmax(dim=1) == labels).sum().item()
    return total_loss / len(train_loader), correct / len(train_loader.dataset)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    video_paths = glob("data/train/*.mp4")
    labels = np.random.randint(0, NUM_CLASSES, len(video_paths))
    dataset = VideoDataset(video_paths, labels, transform)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    model = CNN_LSTM().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    for epoch in range(EPOCHS):
        loss, acc = train(model, train_loader, criterion, optimizer, device)
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {loss:.4f}, Accuracy: {acc:.4f}")
    
    torch.save(model.state_dict(), "cnn_lstm_model.pth")
    print("Model training completed!")

if __name__ == "__main__":
    main()
