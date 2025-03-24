import torch
import torch.nn as nn
import torchvision.models as models
from transformers import AutoImageProcessor, AutoModelForVideoClassification

processor = AutoImageProcessor.from_pretrained("Fulwa/videomae-base1-finetuned-ucf101-subset")
model = AutoModelForVideoClassification.from_pretrained("Fulwa/videomae-base1-finetuned-ucf101-subset")

class CNNFeatureExtractor(nn.Module):
    def __init__(self):
        super(CNNFeatureExtractor, self).__init__()
        base_model = models.resnet18(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(base_model.children())[:-1])
    
    def forward(self, x):
        batch_size, seq_len, c, h, w = x.shape
        x = x.view(batch_size * seq_len, c, h, w)
        features = self.feature_extractor(x)
        features = features.view(batch_size, seq_len, -1)
        return features

class CNN_LSTM(nn.Module):
    def __init__(self, feature_dim=512, hidden_dim=256, num_layers=2, num_classes=5):
        super(CNN_LSTM, self).__init__()
        self.cnn = CNNFeatureExtractor()
        self.lstm = nn.LSTM(input_size=feature_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, dropout=0.3)
        self.fc = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        x = self.cnn(x)
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])
        return x