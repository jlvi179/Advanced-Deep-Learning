from functools import partial
import matplotlib.pyplot as plt
import numpy as np
import sys, os
import typing as tp

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchsummary import summary

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from model_examples import UncertaintyCNN
from helper import get_normalized_data, evaluate_model
from B01train_NN_gaussian_error import createDataLoaders, nll_loss, predictionEvaluation, evaluationPlotting

dblArr: tp.TypeAlias = np.typing.NDArray[np.float64]

DATA_PATH = "../galah4/"
model_name = "CNN_1_gaussian_error"


def main() -> int:
    # Call the function to get normalized data
    spectra, labels, _, n_labels, labelNames, ranges = get_normalized_data(DATA_PATH)  # pyright: ignore[reportAssignmentType]

    # Detect and use Apple Silicon GPU (MPS) if available
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    # device = torch.device("cpu")
    print(f"Using device: {device}")

    # Load the model from the .pth file
    model = UncertaintyCNN(n_labels * 2)
    model.load_state_dict(torch.load(f"models/{model_name}_best.pth", map_location=device, weights_only=True))

    model.to(device)
    # Print the model summary before moving it to the device
    # summary(model, input_size=(1, spectra_length))

    # Prepare data
    _, _, test_loader = createDataLoaders(spectra, labels)

    # Call the function
    loss_function = partial(nll_loss, n_labels=n_labels)

    # evaluate the model predictions
    pred_mean, pred_std, all_true_labels = predictionEvaluation(n_labels, ranges, evaluate_model(model, test_loader, loss_function, device))

    # present results in plots
    evaluationPlotting(all_true_labels, pred_mean, pred_std, labelNames, n_labels)

    return os.EX_OK


if __name__ == "__main__":
    sys.exit(main())
