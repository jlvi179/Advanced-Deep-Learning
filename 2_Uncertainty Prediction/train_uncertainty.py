import os
import sys

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, TensorDataset, random_split
from huggingface_hub import hf_hub_download


current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, current_dir)
sys.path.insert(0, parent_dir)

from helper import denormalize, denormalize_std, get_normalized_data, train_model
from loss_uncertainty import nll_loss
from model_uncertainty import TinyCNNWithUncertainty


DATA_PATH = os.path.abspath(os.path.join(parent_dir, "data", "4"))
learning_rate = 2e-4
batch_size = 32
num_epochs = 100
patience = 10
train_fraction = 0.7
val_fraction = 0.15
test_fraction = 0.15


def ensure_data_downloaded(data_path):
    os.makedirs(data_path, exist_ok=True)
    for filename in ("labels.npy", "spectra.npy"):
        target_file = os.path.join(data_path, filename)
        if os.path.exists(target_file):
            continue
        print(f"Downloading {filename}...")
        hf_hub_download(
            repo_id="simbaswe/galah4",
            filename=filename,
            repo_type="dataset",
            local_dir=data_path,
        )


ensure_data_downloaded(DATA_PATH)


spectra, labels, spectra_length, n_labels, labelNames, ranges = get_normalized_data(DATA_PATH)
print(f"Data loaded: {spectra.shape[0]} samples, {spectra_length} spectra length")
print(f"Labels: {labelNames}")
print(f"Label ranges (normalized): {ranges}")

spectra_tensor = torch.tensor(spectra, dtype=torch.float32)
labels_tensor = torch.tensor(labels, dtype=torch.float32)

total_samples = len(spectra_tensor)
train_size = int(train_fraction * total_samples)
val_size = int(val_fraction * total_samples)
test_size = total_samples - train_size - val_size
print(f"Split: train={train_size}, val={val_size}, test={test_size} (70/15/15)")

train_dataset, val_dataset, test_dataset = random_split(
    TensorDataset(spectra_tensor, labels_tensor),
    [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(42),
)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

device = torch.device("cpu")
print("Using device:", device)

print("Creating model...")
model = TinyCNNWithUncertainty(n_labels).to(device)

loss_function = lambda batch_spectra, batch_labels, model_ref: nll_loss(model_ref(batch_spectra), batch_labels, n_labels)

print("Starting training...")
train_losses, val_losses, best_model_state = train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    loss_function=loss_function,
    learning_rate=learning_rate,
    num_epochs=num_epochs,
    patience=patience,
    device=device,
    model_name=None,
)

if best_model_state is not None:
    model.load_state_dict(best_model_state)

plt.figure(figsize=(10, 5))
plt.plot(train_losses, label="Training Loss", linewidth=2)
plt.plot(val_losses, label="Validation Loss", linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("Loss (NLL)")
plt.title("Training and Validation Loss")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("uncertainty_training_history.png", dpi=150)
plt.show()

print("Evaluating on test set...")
model.eval()
predictions, uncertainties, truths = [], [], []

with torch.no_grad():
    for batch_spectra, batch_labels in test_loader:
        batch_spectra = batch_spectra.to(device).unsqueeze(1)
        means, stds = model.get_predictions_and_uncertainties(batch_spectra)
        predictions.append(means.cpu().numpy())
        uncertainties.append(stds.cpu().numpy())
        truths.append(batch_labels.numpy())

predictions = np.vstack(predictions)
uncertainties = np.vstack(uncertainties)
true_labels = np.vstack(truths)

predictions_denorm = denormalize(predictions, ranges)
true_labels_denorm = denormalize(true_labels, ranges)
uncertainties_denorm = np.column_stack(
    [denormalize_std(uncertainties[:, i], ranges[:, i]) for i in range(n_labels)]
)
residuals = predictions_denorm - true_labels_denorm
standardized_residuals = residuals / uncertainties_denorm

print("\n" + "=" * 60)
print("UNCERTAINTY PREDICTION RESULTS")
print("=" * 60)
for i, label_name in enumerate(labelNames):
    mae = np.mean(np.abs(predictions_denorm[:, i] - true_labels_denorm[:, i]))
    rmse = np.sqrt(np.mean((predictions_denorm[:, i] - true_labels_denorm[:, i]) ** 2))
    print(f"  {label_name:10s}: MAE={mae:.4f}, RMSE={rmse:.4f}, mean σ={np.mean(uncertainties_denorm[:, i]):.4f}")

x = np.linspace(-4, 4, 200)
normal_pdf = np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)

fig, axes = plt.subplots(n_labels, 2, figsize=(14, 4 * n_labels))
if n_labels == 1:
    axes = axes.reshape(1, -1)

for i, label_name in enumerate(labelNames):
    idx = np.argsort(true_labels_denorm[:, i])

    ax = axes[i, 0]
    ax.scatter(true_labels_denorm[idx, i], predictions_denorm[idx, i], alpha=0.5, s=20)
    ax.errorbar(true_labels_denorm[idx, i], predictions_denorm[idx, i], yerr=uncertainties_denorm[idx, i], fmt="none", elinewidth=0.5, alpha=0.3)
    ax.plot([true_labels_denorm[:, i].min(), true_labels_denorm[:, i].max()], [true_labels_denorm[:, i].min(), true_labels_denorm[:, i].max()], "r--", linewidth=2)
    ax.set_title(f"{label_name} - Predictions vs Truth")
    ax.set_xlabel("True " + label_name)
    ax.set_ylabel("Predicted " + label_name)
    ax.grid(True, alpha=0.3)

    ax = axes[i, 1]
    ax.hist(standardized_residuals[:, i], bins=30, alpha=0.7, density=True)
    ax.plot(x, normal_pdf, "r-", linewidth=2)
    ax.set_title(f"{label_name} - Standardized Residuals")
    ax.set_xlabel("Standardized Residuals")
    ax.set_ylabel("Density")
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("uncertainty_predictions_analysis.png", dpi=150)
plt.show()

fig, axes = plt.subplots(1, n_labels, figsize=(5 * n_labels, 4))
if n_labels == 1:
    axes = [axes]

percentiles = np.arange(1, 101) / 100
for i, label_name in enumerate(labelNames):
    abs_residuals = np.abs(residuals[:, i])
    empirical = [np.mean(abs_residuals <= np.percentile(uncertainties_denorm[:, i], p * 100)) for p in percentiles]
    axes[i].plot(percentiles, percentiles, "r--", linewidth=2)
    axes[i].plot(percentiles, empirical, "b-", linewidth=2)
    axes[i].set_title(f"{label_name} - Calibration Plot")
    axes[i].set_xlabel("Expected quantile")
    axes[i].set_ylabel("Empirical coverage")
    axes[i].set_xlim(0, 1)
    axes[i].set_ylim(0, 1)
    axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("uncertainty_calibration.png", dpi=150)
plt.show()

print("Training completed! Check the generated plots for visual analysis.")