import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

from dataset import get_dataloader
from models import ASD_CNN


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc


def validate_epoch(model, test_loader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(test_loader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc, all_preds, all_labels


def train_model(epochs=20, lr=0.001, batch_size=4):
    """Main training function"""
    
    print("=" * 60)
    print("ASD SCREENING - BASELINE CNN TRAINING")
    print("=" * 60)
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")
    
    # Load data with smaller batch size to reduce memory
    print("Loading datasets...")
    from torch.utils.data import DataLoader, random_split
    from dataset import ASD_Dataset
    
    dataset = ASD_Dataset("data/spectrograms")
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print(f"Total images: {len(dataset)}")
    print(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")
    print(f"Batch size: {batch_size}\n")
    
    # Initialize model
    print("Initializing model...")
    model = ASD_CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    print(f"Model initialized with LR={lr}\n")
    
    # Training history
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    best_val_acc = 0.0
    
    print("=" * 65)
    print("STARTING TRAINING...\n")
    print(f"{'Epoch':<6} | {'Train Loss':<12} | {'Train Acc':<10} | {'Val Loss':<12} | {'Val Acc':<10}")
    print("-" * 65)
    
    for epoch in range(epochs):
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validate
        val_loss, val_acc, preds, labels = validate_epoch(model, test_loader, criterion, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        print(f"{epoch+1:<6} | {train_loss:<12.4f} | {train_acc:<10.2f}% | {val_loss:<12.4f} | {val_acc:<10.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "results/asd_model.pth")
            print(f"        [BEST] Model saved (Val Acc: {val_acc:.2f}%)")
    
    print("\n" + "=" * 65)
    print("TRAINING COMPLETE!")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print("=" * 65 + "\n")
    
    # Plot training history
    print("Generating training plots...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss plot
    axes[0].plot(train_losses, label='Train Loss', marker='o', markersize=3)
    axes[0].plot(val_losses, label='Val Loss', marker='s', markersize=3)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training & Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy plot
    axes[1].plot(train_accs, label='Train Acc', marker='o', markersize=3)
    axes[1].plot(val_accs, label='Val Acc', marker='s', markersize=3)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training & Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("results/training_history.png", dpi=150)
    print("[SAVED] results/training_history.png\n")
    
    # Generate final test metrics
    print("Generating final test metrics...")
    model.load_state_dict(torch.load("results/asd_model.pth"))
    model.eval()
    
    final_preds = []
    final_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            final_preds.extend(predicted.cpu().numpy())
            final_labels.extend(labels.cpu().numpy())
    
    # Confusion matrix
    cm = confusion_matrix(final_labels, final_preds)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Autism'],
                yticklabels=['Normal', 'Autism'])
    plt.title('Confusion Matrix - Test Set')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig("results/confusion_matrix.png", dpi=150)
    print("[SAVED] results/confusion_matrix.png\n")
    
    # Classification report
    print("=" * 60)
    print("FINAL TEST SET PERFORMANCE")
    print("=" * 60)
    print(classification_report(final_labels, final_preds, 
                                target_names=['Normal', 'Autism']))
    
    accuracy = accuracy_score(final_labels, final_preds)
    print(f"Overall Test Accuracy: {accuracy*100:.2f}%")
    print("=" * 60 + "\n")
    
    return model, {
        'train_losses': train_losses,
        'train_accs': train_accs,
        'val_losses': val_losses,
        'val_accs': val_accs,
        'best_val_acc': best_val_acc,
        'test_accuracy': accuracy
    }


if __name__ == "__main__":
    model, history = train_model(epochs=20, lr=0.001)
    print("[SUCCESS] All done! Check results/ folder for outputs.")
