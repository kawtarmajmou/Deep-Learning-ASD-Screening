import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from dataset import get_dataloader
from models import ASD_CNN


def _ece_from_bins(y_true, y_prob, n_bins=10):
    """
    Expected Calibration Error (ECE) computed with uniform probability bins.
    y_true: array of 0/1 labels
    y_prob: array of predicted P(class=1)
    """
    y_true = np.asarray(y_true).astype(np.float64)
    y_prob = np.asarray(y_prob).astype(np.float64)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bin_confs = []
    bin_accs = []
    bin_counts = []

    for i in range(n_bins):
        lo = bins[i]
        hi = bins[i + 1]
        # Include the right edge only for the last bin.
        if i < n_bins - 1:
            mask = (y_prob >= lo) & (y_prob < hi)
        else:
            mask = (y_prob >= lo) & (y_prob <= hi)

        if not np.any(mask):
            continue

        bin_count = mask.mean()  # proportion of samples in the bin
        bin_conf = y_prob[mask].mean()
        bin_acc = y_true[mask].mean()  # y_true is 0/1 so mean is empirical prob

        ece += np.abs(bin_acc - bin_conf) * bin_count
        bin_confs.append(bin_conf)
        bin_accs.append(bin_acc)
        bin_counts.append(mask.sum())

    return ece, np.array(bin_confs), np.array(bin_accs), np.array(bin_counts)


def evaluate_mc_dropout(model, test_loader, mc_samples=30, n_bins=10):
    """
    Runs MC-dropout by enabling dropout at inference-time (model.train()) and
    averaging predictions across multiple stochastic forward passes.
    """
    model.train()  # enables dropout; there are no batchnorm layers here

    all_probs_mean = []
    all_labels = []
    all_entropy = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images
            labels = labels

            probs_acc = None
            entropy_acc = None

            for _ in range(mc_samples):
                logits = model(images)
                probs = torch.softmax(logits, dim=1)
                if probs_acc is None:
                    probs_acc = probs
                else:
                    probs_acc += probs

            probs_mean = probs_acc / mc_samples
            entropy = -(probs_mean * torch.log(probs_mean + 1e-12)).sum(dim=1)

            all_probs_mean.append(probs_mean.cpu())
            all_labels.append(labels.cpu())
            all_entropy.append(entropy.cpu())

    probs_mean = torch.cat(all_probs_mean, dim=0)
    labels_all = torch.cat(all_labels, dim=0)
    entropy_all = torch.cat(all_entropy, dim=0).numpy()

    preds = torch.argmax(probs_mean, dim=1).numpy()
    y_true = labels_all.numpy()

    # Reliability diagram uses P(class=1)
    y_prob = probs_mean[:, 1].numpy()

    acc = accuracy_score(y_true, preds)
    ece, bin_confs, bin_accs, bin_counts = _ece_from_bins(
        y_true=y_true, y_prob=y_prob, n_bins=n_bins
    )

    return {
        "accuracy": acc,
        "ece": float(ece),
        "y_true": y_true,
        "y_prob": y_prob,
        "preds": preds,
        "entropy": entropy_all,
        "bin_confs": bin_confs,
        "bin_accs": bin_accs,
        "bin_counts": bin_counts,
    }


def evaluate_model():
    # Load model
    model = ASD_CNN()
    model.load_state_dict(torch.load("results/asd_model.pth"))
    model.eval()

    # Load test data
    test_loader = get_dataloader(train=False)

    # --------------------
    # Baseline evaluation
    # --------------------
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())

    # Accuracy
    acc = accuracy_score(all_labels, all_preds)
    print(f"\n Accuracy: {acc:.4f}")

    # Report
    print("\n Classification Report:")
    print(classification_report(all_labels, all_preds))

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)

    plt.figure()
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig("results/confusion_matrix.png")
    plt.show()

    # --------------------
    # MC-dropout evaluation
    # --------------------
    print("\nRunning MC-dropout reliability evaluation...")
    mc_out = evaluate_mc_dropout(model, test_loader, mc_samples=20, n_bins=10)
    print(f"MC-dropout Accuracy: {mc_out['accuracy']:.4f}")
    print(f"MC-dropout ECE: {mc_out['ece']:.4f}")

    # Confusion matrix for MC-dropout predictions
    cm_mc = confusion_matrix(mc_out["y_true"], mc_out["preds"])
    plt.figure()
    sns.heatmap(cm_mc, annot=True, fmt="d")
    plt.title("Confusion Matrix (MC-dropout)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig("results/confusion_matrix_mc.png")
    plt.show()

    # Reliability diagram
    plt.figure(figsize=(5, 5))
    plt.plot(mc_out["bin_confs"], mc_out["bin_accs"], marker="o", label="Model")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly calibrated")
    plt.title(f"Reliability Diagram (ECE={mc_out['ece']:.3f})")
    plt.xlabel("Mean predicted P(class=1)")
    plt.ylabel("Empirical P(class=1)")
    plt.legend()
    plt.savefig("results/reliability_diagram.png")
    plt.show()

    # Uncertainty vs correctness (entropy)
    correct_mask = (mc_out["preds"] == mc_out["y_true"])
    plt.figure(figsize=(6, 4))
    plt.hist(mc_out["entropy"][correct_mask], bins=40, alpha=0.6, label="Correct")
    plt.hist(mc_out["entropy"][~correct_mask], bins=40, alpha=0.6, label="Incorrect")
    plt.title("Predictive Entropy Distribution (MC-dropout)")
    plt.xlabel("Entropy")
    plt.ylabel("Count")
    plt.legend()
    plt.savefig("results/entropy_hist.png")
    plt.show()

    with open("results/ece.txt", "w", encoding="utf-8") as f:
        f.write(f"MC-dropout ECE: {mc_out['ece']:.6f}\n")
        f.write(f"MC-dropout Accuracy: {mc_out['accuracy']:.6f}\n")


if __name__ == "__main__":
    evaluate_model()