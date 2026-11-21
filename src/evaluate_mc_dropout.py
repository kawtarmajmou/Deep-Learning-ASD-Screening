import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
from torch.utils.data import DataLoader, random_split

from dataset import ASD_Dataset
from models import ASD_CNN


def mc_dropout_evaluation(model, test_loader, mc_samples=30, device='cpu'):
    """
    Perform MC-dropout evaluation for uncertainty estimation.
    This runs inference with dropout enabled multiple times and averages predictions.
    """
    model.train()  # Enable dropout
    
    all_probs_mean = []
    all_probs_std = []
    all_labels = []
    all_entropy = []
    all_predictions = []
    
    print("\nPerforming MC-Dropout uncertainty estimation...")
    print(f"MC samples: {mc_samples}\n")
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            images = images.to(device)
            labels = labels.cpu().numpy()
            
            # Collect multiple stochastic forward passes
            batch_probs = []
            
            for sample in range(mc_samples):
                logits = model(images)
                probs = torch.softmax(logits, dim=1)
                batch_probs.append(probs.cpu().numpy())
            
            # Stack and compute statistics
            batch_probs = np.array(batch_probs)  # (mc_samples, batch_size, 2)
            probs_mean = batch_probs.mean(axis=0)  # (batch_size, 2)
            probs_std = batch_probs.std(axis=0)    # (batch_size, 2)
            
            # Get predictions from mean probabilities
            preds = np.argmax(probs_mean, axis=1)
            
            # Calculate entropy
            entropy = -(probs_mean * np.log(probs_mean + 1e-10)).sum(axis=1)
            
            all_probs_mean.append(probs_mean)
            all_probs_std.append(probs_std)
            all_predictions.append(preds)
            all_labels.append(labels)
            all_entropy.append(entropy)
            
            if (batch_idx + 1) % 50 == 0:
                print(f"  Processed {(batch_idx + 1) * len(images)} samples...")
    
    # Concatenate all batches
    probs_mean = np.concatenate(all_probs_mean, axis=0)
    probs_std = np.concatenate(all_probs_std, axis=0)
    predictions = np.concatenate(all_predictions, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    entropy = np.concatenate(all_entropy, axis=0)
    
    # Compute metrics
    accuracy = accuracy_score(labels, predictions)
    
    return {
        'accuracy': accuracy,
        'probs_mean': probs_mean,
        'probs_std': probs_std,
        'entropy': entropy,
        'predictions': predictions,
        'labels': labels
    }


def generate_calibration_plot(probs_mean, labels, n_bins=10):
    """Generate reliability/calibration curve"""
    # Use probability of positive class
    y_prob = probs_mean[:, 1]
    
    # Bin the predictions
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    bin_accs = []
    bin_confs = []
    bin_counts = []
    
    for i in range(n_bins):
        mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i+1])
        if i == n_bins - 1:
            mask = (y_prob >= bin_edges[i]) & (y_prob <= bin_edges[i+1])
        
        if mask.sum() > 0:
            bin_acc = labels[mask].mean()
            bin_conf = y_prob[mask].mean()
            bin_count = mask.sum()
            
            bin_accs.append(bin_acc)
            bin_confs.append(bin_conf)
            bin_counts.append(bin_count)
    
    # Calculate ECE (Expected Calibration Error)
    ece = 0
    for conf, acc, count in zip(bin_confs, bin_accs, bin_counts):
        ece += np.abs(conf - acc) * (count / len(labels))
    
    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', linewidth=2)
    
    # Actual calibration
    if bin_confs:
        ax.plot(bin_confs, bin_accs, 'o-', label='Model Calibration', linewidth=2, markersize=8)
    
    ax.set_xlabel('Mean Predicted Probability', fontsize=12)
    ax.set_ylabel('Actual Accuracy', fontsize=12)
    ax.set_title(f'Reliability Diagram (ECE: {ece:.4f})', fontsize=13)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    
    return fig, ece


def generate_uncertainty_plots(results):
    """Generate plots for uncertainty analysis"""
    entropy = results['entropy']
    predictions = results['predictions']
    labels = results['labels']
    
    # Separate correct and incorrect predictions
    correct_mask = (predictions == labels)
    
    correct_entropy = entropy[correct_mask]
    incorrect_entropy = entropy[~correct_mask]
    
    # Entropy distribution plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].hist(correct_entropy, bins=30, alpha=0.6, label='Correct Predictions', color='green', edgecolor='black')
    axes[0].hist(incorrect_entropy, bins=30, alpha=0.6, label='Incorrect Predictions', color='red', edgecolor='black')
    axes[0].set_xlabel('Prediction Entropy', fontsize=11)
    axes[0].set_ylabel('Frequency', fontsize=11)
    axes[0].set_title('Entropy Distribution by Prediction Correctness', fontsize=12)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Uncertainty vs Accuracy by threshold
    thresholds = np.linspace(entropy.min(), entropy.max(), 20)
    accuracies_by_threshold = []
    
    for threshold in thresholds:
        mask = entropy <= threshold
        if mask.sum() > 0:
            acc = accuracy_score(labels[mask], predictions[mask])
            accuracies_by_threshold.append(acc)
        else:
            accuracies_by_threshold.append(0)
    
    axes[1].plot(thresholds, accuracies_by_threshold, 'o-', linewidth=2, markersize=6)
    axes[1].set_xlabel('Entropy Threshold', fontsize=11)
    axes[1].set_ylabel('Accuracy (Samples Below Threshold)', fontsize=11)
    axes[1].set_title('Accuracy vs Uncertainty Threshold', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def main_evaluation(model_path="results/asd_model.pth", mc_samples=30):
    """Main evaluation function"""
    
    print("="*70)
    print("MC-DROPOUT UNCERTAINTY EVALUATION")
    print("="*70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}\n")
    
    # Load model
    print("Loading model...")
    model = ASD_CNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    print("[SUCCESS] Model loaded\n")
    
    # Load test data
    print("Loading test dataset...")
    dataset = ASD_Dataset("data/spectrograms")
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=0)
    print(f"Test samples: {len(test_dataset)}\n")
    
    # MC-Dropout evaluation
    results = mc_dropout_evaluation(model, test_loader, mc_samples=mc_samples, device=device)
    
    print(f"\nMC-Dropout Test Accuracy: {results['accuracy']*100:.2f}%")
    
    # Generate plots
    print("\nGenerating evaluation plots...")
    
    # 1. Calibration plot
    fig, ece = generate_calibration_plot(results['probs_mean'], results['labels'])
    plt.savefig("results/calibration_curve.png", dpi=150, bbox_inches='tight')
    print("[SAVED] results/calibration_curve.png")
    plt.close()
    
    # 2. Uncertainty plots
    fig = generate_uncertainty_plots(results)
    plt.savefig("results/uncertainty_analysis.png", dpi=150, bbox_inches='tight')
    print("[SAVED] results/uncertainty_analysis.png")
    plt.close()
    
    # 3. Std deviation by class
    fig, ax = plt.subplots(figsize=(10, 6))
    
    std_mean_class0 = results['probs_std'][results['labels']==0, 0].mean()
    std_mean_class1 = results['probs_std'][results['labels']==1, 0].mean()
    
    classes = ['Normal', 'Autism']
    stds = [std_mean_class0, std_mean_class1]
    colors = ['#1f77b4', '#ff7f0e']
    
    ax.bar(classes, stds, color=colors, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Mean Prediction Std Dev', fontsize=11)
    ax.set_title('Prediction Uncertainty by Class', fontsize=12)
    
    for i, v in enumerate(stds):
        ax.text(i, v + 0.005, f'{v:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig("results/std_by_class.png", dpi=150, bbox_inches='tight')
    print("[SAVED] results/std_by_class.png")
    plt.close()
    
    # Print detailed report
    print("\n" + "="*70)
    print("DETAILED EVALUATION REPORT")
    print("="*70)
    print(f"\nAccuracy: {results['accuracy']*100:.2f}%")
    print(f"ECE (Expected Calibration Error): {ece:.4f}")
    print(f"\nEntropy Statistics:")
    print(f"  Mean: {results['entropy'].mean():.4f}")
    print(f"  Std:  {results['entropy'].std():.4f}")
    print(f"  Min:  {results['entropy'].min():.4f}")
    print(f"  Max:  {results['entropy'].max():.4f}")
    
    print(f"\nPrediction Std Dev by Class:")
    print(f"  Normal (Class 0): {std_mean_class0:.4f}")
    print(f"  Autism (Class 1): {std_mean_class1:.4f}")
    
    print("\n" + "="*70)
    print("CLASSIFICATION REPORT")
    print("="*70)
    print(classification_report(results['labels'], results['predictions'],
                               target_names=['Normal', 'Autism']))
    
    print("\n[SUCCESS] Evaluation complete!")


if __name__ == "__main__":
    main_evaluation(model_path="results/asd_model.pth", mc_samples=30)
