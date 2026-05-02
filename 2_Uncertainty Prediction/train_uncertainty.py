import os
import sys
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
from matplotlib import pyplot as plt

# Add parent directories to path to import helper and local modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, current_dir)
sys.path.insert(0, parent_dir)

from helper import normalize, denormalize, denormalize_std, train_model, get_normalized_data
from model_uncertainty import TinyCNNWithUncertainty
from loss_uncertainty import nll_loss

# Data path
DATA_PATH = "../../data/4/"

# Hyperparameters
learning_rate = 2e-4
batch_size = 32
num_epochs = 100
patience = 10
train_fraction = 0.7
val_fraction = 0.15

def _load_local_labels():
    for path in [
        "labels.npy",
        "../1_Astronomy CNN with PyTorch/labels.npy",
        os.path.join(parent_dir, "1_Astronomy CNN with PyTorch", "labels.npy"),
    ]:
        if os.path.exists(path):
            labels_raw = np.load(path)
            print(f"Loaded labels from {path}: {labels_raw.shape}")
            return labels_raw
    raise FileNotFoundError("Could not find labels.npy in any expected location")


def _create_synthetic_data(labels_raw):
    snr = labels_raw[:, -1]
    labels, ranges = normalize(labels_raw[:, :-1][:, -3:], 0.05)
    label_names = ["t_eff", "log_g", "fe_h"]

    spectra_length = 3400
    n_samples = labels.shape[0]
    print(f"Generating {n_samples} synthetic spectra of length {spectra_length}...")

    wavelength = np.linspace(4700, 7700, spectra_length)
    continuum = (0.5 + 0.3 * labels[:, 0])[:, None]
    feature_scale = (0.1 * (1 - labels[:, 1]) * (1 + labels[:, 2]))[:, None]
    features = feature_scale * np.sin(wavelength / 200)[None, :]
    noise_level = np.where(snr > 0, 1.0 / snr, 0.1)[:, None]
    noise = noise_level * np.random.randn(n_samples, spectra_length)
    spectra = np.log(np.clip(continuum + features + noise, 0.1, 10.0))

    print(f"Synthetic data created: {n_samples} samples, {spectra_length} spectra length")
    return spectra, labels, spectra_length, labels.shape[1], label_names, ranges


def load_data():
    print("Loading data...")
    try:
        data = get_normalized_data(DATA_PATH)
        print(f"Data loaded: {data[0].shape[0]} samples, {data[2]} spectra length")
        return data
    except FileNotFoundError as e:
        print(f"Error loading data from {DATA_PATH}: {e}")
        print("\nCreating synthetic data for demonstration...")
        try:
            return _create_synthetic_data(_load_local_labels())
        except FileNotFoundError:
            raise RuntimeError("Neither real data path nor local labels.npy found. Please ensure data is available.")


def create_loaders(spectra, labels):
    spectra_tensor = torch.tensor(spectra, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.float32)

    print(f"\nSplitting data: train={train_fraction}, val={val_fraction}, test={(1 - train_fraction - val_fraction)}")
    total = len(spectra_tensor)
    train_size = int(train_fraction * total)
    val_size = int(val_fraction * total)
    test_size = total - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        TensorDataset(spectra_tensor, labels_tensor),
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42),
    )
    print(f"Train: {train_size}, Val: {val_size}, Test: {test_size}")
    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        DataLoader(val_dataset, batch_size=batch_size, shuffle=False),
        DataLoader(test_dataset, batch_size=batch_size, shuffle=False),
    )


def pick_device():
    try:
        import torch_xpu as _tx

        has_xpu = True
    except Exception:
        has_xpu = False

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif has_xpu:
        device = torch.device("xpu")
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print("Using device:", device)
    return device


def evaluate_on_test(model, test_loader, device, ranges, n_labels):
    print("\nEvaluating on test set...")
    model.eval()
    preds, uncs, truths = [], [], []

    with torch.no_grad():
        for batch_spectra, batch_labels in test_loader:
            batch_spectra = batch_spectra.to(device).unsqueeze(1)
            means, stds = model.get_predictions_and_uncertainties(batch_spectra)
            preds.append(means.cpu().numpy())
            uncs.append(stds.cpu().numpy())
            truths.append(batch_labels.numpy())

    predictions = np.vstack(preds)
    uncertainties = np.vstack(uncs)
    true_labels = np.vstack(truths)

    predictions_denorm = denormalize(predictions, ranges)
    true_labels_denorm = denormalize(true_labels, ranges)
    uncertainties_denorm = np.column_stack(
        [denormalize_std(uncertainties[:, i], ranges[:, i]) for i in range(n_labels)]
    )

    residuals = predictions_denorm - true_labels_denorm
    standardized_residuals = residuals / uncertainties_denorm
    nll_values = 0.5 * np.log(2 * np.pi * uncertainties_denorm**2) + 0.5 * (residuals / uncertainties_denorm) ** 2

    return predictions_denorm, uncertainties_denorm, true_labels_denorm, residuals, standardized_residuals, nll_values


def print_metrics(label_names, predictions_denorm, uncertainties_denorm, true_labels_denorm, residuals, standardized_residuals, nll_values):
    print("\n" + "=" * 60)
    print("UNCERTAINTY PREDICTION RESULTS")
    print("=" * 60)

    print("\nPrediction accuracy (denormalized):")
    for i, label_name in enumerate(label_names):
        mae = np.mean(np.abs(predictions_denorm[:, i] - true_labels_denorm[:, i]))
        rmse = np.sqrt(np.mean((predictions_denorm[:, i] - true_labels_denorm[:, i]) ** 2))
        print(f"  {label_name:10s}: MAE={mae:.4f}, RMSE={rmse:.4f}")

    print("\nPredicted uncertainties (denormalized, mean±std):")
    for i, label_name in enumerate(label_names):
        print(f"  {label_name:10s}: {np.mean(uncertainties_denorm[:, i]):.4f} ± {np.std(uncertainties_denorm[:, i]):.4f}")

    print("\n" + "-" * 60)
    print("UNCERTAINTY CALIBRATION ANALYSIS")
    print("-" * 60)

    print("\nStandardized residuals statistics (should be ~N(0,1) if calibrated):")
    for i, label_name in enumerate(label_names):
        print(f"  {label_name:10s}: mean={np.mean(standardized_residuals[:, i]):.4f}, std={np.std(standardized_residuals[:, i]):.4f}")

    print("\nPrediction interval coverage (fraction of true values within ±1σ, ±2σ, ±3σ):")
    for i, label_name in enumerate(label_names):
        c1 = np.mean(np.abs(residuals[:, i]) <= uncertainties_denorm[:, i])
        c2 = np.mean(np.abs(residuals[:, i]) <= 2 * uncertainties_denorm[:, i])
        c3 = np.mean(np.abs(residuals[:, i]) <= 3 * uncertainties_denorm[:, i])
        print(f"  {label_name:10s}: ±1σ={c1:.1%}, ±2σ={c2:.1%}, ±3σ={c3:.1%}")
        print("              (Expected: ±1σ≈68%, ±2σ≈95%, ±3σ≈99.7%)")

    print("\nTest set statistics:")
    print(f"  Mean NLL: {np.mean(nll_values):.4f}")
    print(f"  Mean NLL per label: {np.mean(nll_values, axis=0)}")


def plot_training_history(train_losses, val_losses):
    print("\nPlotting training history...")
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
    print("Saved: uncertainty_training_history.png")


def plot_prediction_analysis(label_names, predictions_denorm, uncertainties_denorm, true_labels_denorm, standardized_residuals):
    print("\nCreating visualization plots...")
    fig, axes = plt.subplots(len(label_names), 2, figsize=(14, 4 * len(label_names)))
    if len(label_names) == 1:
        axes = axes.reshape(1, -1)

    from scipy.stats import norm

    for i, label_name in enumerate(label_names):
        ax = axes[i, 0]
        sorted_idx = np.argsort(true_labels_denorm[:, i])
        x_true = true_labels_denorm[sorted_idx, i]
        y_pred = predictions_denorm[sorted_idx, i]
        y_std = uncertainties_denorm[sorted_idx, i]
        ax.scatter(x_true, y_pred, alpha=0.5, s=20, label="Predictions")
        ax.errorbar(x_true, y_pred, yerr=y_std, fmt="none", elinewidth=0.5, alpha=0.3)
        ax.plot([x_true.min(), x_true.max()], [x_true.min(), x_true.max()], "r--", linewidth=2, label="Perfect prediction")
        ax.set_xlabel("True " + label_name)
        ax.set_ylabel("Predicted " + label_name)
        ax.set_title(f"{label_name} - Predictions vs Truth")
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[i, 1]
        ax.hist(standardized_residuals[:, i], bins=30, alpha=0.7, label="Data", density=True)
        x = np.linspace(-4, 4, 100)
        ax.plot(x, norm.pdf(x), "r-", linewidth=2, label="Standard Normal")
        ax.set_xlabel("Standardized Residuals")
        ax.set_ylabel("Density")
        ax.set_title(f"{label_name} - Standardized Residuals")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("uncertainty_predictions_analysis.png", dpi=150)
    plt.show()
    print("Saved: uncertainty_predictions_analysis.png")


def plot_calibration(label_names, residuals, uncertainties_denorm):
    fig, axes = plt.subplots(1, len(label_names), figsize=(5 * len(label_names), 4))
    if len(label_names) == 1:
        axes = [axes]

    percentiles = np.arange(1, 101)
    expected = percentiles / 100
    for i, label_name in enumerate(label_names):
        abs_residuals = np.abs(residuals[:, i])
        empirical = [np.mean(abs_residuals <= np.percentile(uncertainties_denorm[:, i], p)) for p in percentiles]
        ax = axes[i]
        ax.plot(expected, expected, "r--", linewidth=2, label="Perfect calibration")
        ax.plot(expected, empirical, "b-", linewidth=2, label="Model")
        ax.set_xlabel("Expected quantile")
        ax.set_ylabel("Empirical coverage")
        ax.set_title(f"{label_name} - Calibration Plot")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig("uncertainty_calibration.png", dpi=150)
    plt.show()
    print("Saved: uncertainty_calibration.png")


def main():
    spectra, labels, spectra_length, n_labels, label_names, ranges = load_data()
    print(f"Labels: {label_names}")
    print(f"Label ranges (normalized): {ranges}")

    train_loader, val_loader, test_loader = create_loaders(spectra, labels)
    device = pick_device()

    print("\nCreating model...")
    model = TinyCNNWithUncertainty(n_labels).to(device)
    loss_function = lambda batch_spectra, batch_labels, model_ref: nll_loss(model_ref(batch_spectra), batch_labels, n_labels)

    print("\nStarting training...")
    train_losses, val_losses, best_model_state = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_function=loss_function,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        patience=patience,
        device=device,
        model_name="uncertainty_cnn",
    )
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    plot_training_history(train_losses, val_losses)

    results = evaluate_on_test(model, test_loader, device, ranges, n_labels)
    predictions_denorm, uncertainties_denorm, true_labels_denorm, residuals, standardized_residuals, nll_values = results
    print_metrics(label_names, predictions_denorm, uncertainties_denorm, true_labels_denorm, residuals, standardized_residuals, nll_values)
    plot_prediction_analysis(label_names, predictions_denorm, uncertainties_denorm, true_labels_denorm, standardized_residuals)
    plot_calibration(label_names, residuals, uncertainties_denorm)

    print("\n" + "=" * 60)
    print("Training completed! Check the generated plots for visual analysis.")
    print("=" * 60)


if __name__ == "__main__":
    main()
