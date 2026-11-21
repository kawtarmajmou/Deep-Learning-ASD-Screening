import torch
import torch.nn as nn
import torchvision.models as models


class ASD_CNN(nn.Module):
    """Baseline CNN for ASD classification from spectrogram images"""
    def __init__(self):
        super(ASD_CNN, self).__init__()

        # Input is 3-channel (RGB spectrogram images)
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.AdaptiveAvgPool2d((16, 16))  # Adaptive pooling for consistent output size
        )

        # Fixed flattened size: 64 channels * 16 * 16 = 16384
        self.fc = nn.Sequential(
            nn.Linear(64 * 16 * 16, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(128, 2)
        )

        # Dropout for MC-dropout reliability estimation
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class ASD_TransferLearning_MobileNetV2(nn.Module):
    """Transfer Learning using MobileNetV2 for lightweight inference"""
    def __init__(self, pretrained=True):
        super(ASD_TransferLearning_MobileNetV2, self).__init__()
        
        # Load pretrained MobileNetV2 using newer API
        if pretrained:
            from torchvision.models import MobileNet_V2_Weights
            self.backbone = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V2)
        else:
            self.backbone = models.mobilenet_v2(weights=None)
        
        # Freeze backbone layers
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Replace classifier head
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(128, 2)
        )
        
        # Unfreeze last few layers for fine-tuning
        for param in self.backbone.features[-3:].parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.backbone(x)


class ASD_TransferLearning_ResNet18(nn.Module):
    """Transfer Learning using ResNet18 for better feature extraction"""
    def __init__(self, pretrained=True):
        super(ASD_TransferLearning_ResNet18, self).__init__()
        
        # Load pretrained ResNet18 using newer API
        if pretrained:
            from torchvision.models import ResNet18_Weights
            self.backbone = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        else:
            self.backbone = models.resnet18(weights=None)
        
        # Freeze backbone layers
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Replace classifier head
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(128, 2)
        )
        
        # Unfreeze last few layers for fine-tuning
        for param in self.backbone.layer4.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.backbone(x)


class CNN_LSTM_Hybrid(nn.Module):
    """Hybrid CNN-LSTM for temporal-spectral feature extraction"""
    def __init__(self, seq_length=5):
        super(CNN_LSTM_Hybrid, self).__init__()
        
        self.seq_length = seq_length
        
        # CNN for spatial-frequency feature extraction from each spectrogram
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.AdaptiveAvgPool2d((8, 8))
        )
        
        # Flatten CNN output: 64 * 8 * 8 = 4096
        cnn_output_size = 64 * 8 * 8
        
        # LSTM for temporal modeling across spectrogram sequence
        self.lstm = nn.LSTM(
            input_size=cnn_output_size,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            dropout=0.5
        )
        
        # Classifier head
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(128, 2)
        )
        
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        # x shape: (batch, seq_length, channels, height, width)
        batch_size, seq_len, c, h, w = x.shape
        
        # Process each spectrogram through CNN
        cnn_features = []
        for t in range(seq_len):
            frame = x[:, t, :, :, :]  # (batch, channels, height, width)
            feat = self.cnn(frame)
            feat = feat.view(batch_size, -1)  # Flatten
            cnn_features.append(feat)
        
        # Stack CNN features: (batch, seq_length, cnn_output_size)
        cnn_output = torch.stack(cnn_features, dim=1)
        
        # Pass through LSTM
        lstm_output, (h_n, c_n) = self.lstm(cnn_output)
        
        # Use last hidden state
        last_hidden = h_n[-1]  # (batch, hidden_size)
        
        # Classify
        output = self.fc(last_hidden)
        return output


def train_model():
    train_loader = get_dataloader(train=True)

    model = ASD_CNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    losses = []

    for epoch in range(10):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        losses.append(epoch_loss)

        print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}")

    # Save model
    torch.save(model.state_dict(), "results/asd_model.pth")

    # Plot loss
    plt.plot(losses)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig("results/loss_curve.png")
    plt.show()

    print("Training Complete!")


if __name__ == "__main__":
    train_model()