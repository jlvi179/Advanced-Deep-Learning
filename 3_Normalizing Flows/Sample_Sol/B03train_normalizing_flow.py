import argparse
from functools import partial
import glob
import jammy_flows
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import subprocess
import sys, os
import typing as tp

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from model_examples import TinyCNNEncoder
from helper import denormalize, denormalize_std, train_model, get_normalized_data, evaluate_model

dblArr: tp.TypeAlias = np.typing.NDArray[np.float64]

DATA_PATH = "../galah4/"

# Hyperparameters
learning_rate = 0.8e-5
batch_size = 32
num_epochs = 400
patience = 30  # Training loop with early stopping, if the validation loss does not improve for 'patience' epochs
train_fraction = 0.7  # Fraction of the data used for training
val_fraction = 0.15  # Fraction of the data used for validation


class CombinedModel(nn.Module):
    """
    A combined model that integrates a normalizing flow with a CNN encoder.

    Parameters
    ----------
    encoder : torch.nn.Module
        The encoder model that converts input data to the latent space, i.e., the parameters of the normalizing flow.
    fp64_on_cpu: bool
    nf_type : str, optional
        The type of normalizing flow to use. Options are "diagonal_gaussian", "full_gaussian", and "full_flow". Default is "diagonal_gaussian".
    Methods
    -------
    log_pdf_evaluation(target_labels, input_data)
        Evaluates the log probability density function for the given target labels and input data.
    sample(flow_params, samplesize_per_batchitem=10000)
        Samples new points from the PDF given input data.
    forward(input_data)
        Performs a forward pass through the model, predicting the mean and standard deviation of the samples.
    visualize_pdf(input_data, filename, samplesize=10000, batch_index=0, truth=None)
        Visualizes the PDF by plotting histograms of the samples and their Gaussian approximations.
    """

    def __init__(self, encoder: type[nn.Module], fp64_on_cpu: bool = False, nf_type: str = "diagonal_gaussian") -> None:
        super().__init__()

        # we define a 3-d PDF over Euclidean spae (e3)
        # using recommended settings (https://github.com/thoglu/jammy_flows/issues/5 scroll down)
        opt_dict = {}
        opt_dict["t"] = {}
        if nf_type == "diagonal_gaussian":
            opt_dict["t"]["cov_type"] = "diagonal"
            flow_defs = "t"
        elif nf_type == "full_gaussian":
            opt_dict["t"]["cov_type"] = "full"
            flow_defs = "t"
        elif nf_type == "full_flow":
            opt_dict["t"]["cov_type"] = "full"
            flow_defs = "gggt"
        else:
            raise Exception("Unknown nf type ", nf_type)

        opt_dict["g"] = dict()
        opt_dict["g"]["fit_normalization"] = 1
        opt_dict["g"]["upper_bound_for_widths"] = 1.0
        opt_dict["g"]["lower_bound_for_widths"] = 0.01

        self.nf_type = nf_type

        # 3d PDF (e3) with gggt flow structure. Four Gaussianization-flow (https://arxiv.org/abs/2003.01941) layers ("g") and an affine flow ("t")
        self.pdf = jammy_flows.pdf("e3", flow_defs, options_overwrite=opt_dict, amortize_everything=True, amortization_mlp_use_custom_mode=True)

        # get the number of flow parameters
        num_flow_parameters = self.pdf.total_number_amortizable_params

        print("The normalizing flow has ", num_flow_parameters, " parameters...")

        # latent dimension (output of the CNN encoder) is set to 128
        self.encoder = encoder(num_flow_parameters)

        # set full precision availability variable as member
        self.fp64_on_cpu = fp64_on_cpu

    def log_pdf_evaluation(self, target_labels: torch.Tensor, input_data: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the log probability density function (PDF) for the given target labels and input data.

        The normalizing flow parameters are predicted by the encoder network based on the input data.
        Then, the log PDF is evaluated at the position of the label.

        Parameters:
        -----------
        target_labels : torch.Tensor
            The target labels for which the log PDF is to be evaluated.
        input_data : torch.Tensor
            The input data to be encoded and used for evaluating the log PDF.
        Returns:
        --------
        log_pdf : torch.Tensor
            The evaluated log PDF for the given target labels and input data.
        """
        latent_intermediate = self.encoder(input_data)  # get the flow parameters from the CNN encoder

        if self.nf_type == "full_flow":
            # convert to double. Double precision is needed for the Gaussianization flow. This is for numerical stability.
            if self.fp64_on_cpu is True:  # MPS does not support double precision, therefore we need to run the flow on the CPU
                latent_intermediate = latent_intermediate.cpu().to(torch.float64)
                target_labels = target_labels.cpu().to(torch.float64)
            else:
                latent_intermediate = latent_intermediate.to(torch.float64)
                target_labels = target_labels.to(torch.float64)

        # evaluate the log PDF at the target labels
        log_pdf, _, _ = self.pdf(target_labels, amortization_parameters=latent_intermediate)
        return log_pdf

    def sample(self, flow_params: torch.Tensor, samplesize_per_batchitem: int = 1000) -> torch.Tensor:
        """
        Sample new points from the PDF given input data.

        Parameters
        ----------
        flow_params : tensor
            Parameters for the normalizing flow, must be of shape (B, L) where B is the batch size and L is the latent dimension.
        samplesize_per_batchitem : int, optional
            Number of samples to draw per batch item. Defaults to 1000.

        Returns
        -------
        tensor
            A tensor of shape (B, S, D) where B is the batch dimension, S is the number of samples,
            and D is the dimension of the target space for the samples.
        """
        # for full flow we need to convert to double precision for the normalizing flow
        # for numerical stability
        if self.nf_type == "full_flow":
            # convert to double
            if self.fp64_on_cpu:  # MPS does not support double precision, therefore we need to run the flow on the CPU
                flow_params = flow_params.cpu().to(torch.float64)
            else:
                flow_params = flow_params.to(torch.float64)

        batch_size = flow_params.shape[0]  # get the batch size
        # sample from the normalizing flow
        repeated_samples, _, _, _ = self.pdf.sample(
            amortization_parameters=flow_params.repeat_interleave(samplesize_per_batchitem, dim=0), allow_gradients=False
        )

        # reshape the samples to be grouped by batch item
        reshaped_samples = repeated_samples[:, None, :].view(batch_size, samplesize_per_batchitem, -1)

        return reshaped_samples

    def forward(self, input_data: torch.Tensor, samplesize_per_batchitem: int = 1000) -> torch.Tensor:
        """
        Perform a forward pass through the model, predicting the mean and standard deviation of the samples.

        Normalizing flows do not directly predict the target labels. Instead, they predict the parameters of the flow that
        transforms the base distribution to the target distribution. Often, we still want to predict the target labels.
        Then, we can sample from the distribution and form the mean of the samples and their standard deviations.
        This is what this function does.

        Parameters
        ----------
        input_data : torch.Tensor
            The input data tensor.
        samplesize_per_batchitem : int, optional
            The number of samples to draw per batch item. Defaults to 1000.

        Returns
        -------
        torch.Tensor
            A tensor of size (B, D*2) where the first half (size D) are the means,
            the second half (another D) are the standard deviations.
        """
        flow_params = self.encoder(input_data)
        samples = self.sample(flow_params, samplesize_per_batchitem=samplesize_per_batchitem)

        # form mean along dim 1 (samples)
        means = samples.mean(dim=1)
        # form std along dim 1 (samples)
        std_deviations = samples.std(dim=1)

        # return means and std deviations as a concatenated tensor along dim 1
        return torch.cat([means, std_deviations], dim=1)

    def visualize_pdf(self, input_data: torch.Tensor, filename: str, sample_size: int = 1000, batch_index: int = 0, truth: torch.Tensor | None = None) -> None:
        """
        Visualizes the probability density function (PDF) of the given input data using a normalizing flow model.

        The function generates samples from the normalizing flow (using the sample() function)
        and plots the histogram of the samples together with a Gaussian approximation.

        Parameters
        ----------
        input_data : torch.Tensor
            The input data tensor from which to pick one batch item for visualization.
        filename : str
            The filename where the resulting plot will be saved.
        sample_size : int, optional
            The number of samples to generate for the PDF visualization (default is 10000).
        batch_index : int, optional
            The index of the batch item to visualize (default is 0).
        truth : torch.Tensor, optional
            The true values of the labels, used for comparison in the plot (default is None).

        Returns
        -------
        None
        """
        # pick out one input from batch
        input_b_item = input_data[batch_index : batch_index + 1]

        # get the flow parameters (by passing the input data through the CNN encoder network)
        flow_params = self.encoder(input_b_item)

        # sample from the normalizing flow (i.e. samples are drawn from the base distribution and transformed by the flow
        # using the change-of-variable formula)
        samples = self.sample(flow_params, samplesize_per_batchitem=sample_size)
        # the rest of the code is just plotting.

        # we only have 1 batch item
        samples = samples.squeeze(0)

        # plot three 1-dimensional distributions together with normal approximation,
        # so we calculate the mean and std of the samples
        mean = samples.mean(dim=0).cpu().numpy()
        std = samples.std(dim=0).cpu().numpy()
        samples = samples.cpu().numpy()

        fig, ax_dict = plt.subplots(3, 1)
        for dim_ind in range(3):
            # plot the histogram of the samples
            ax_dict[dim_ind].hist(samples[:, dim_ind], color="k", density=True, bins=50, alpha=0.5, label="density based on samples")

            # plot the Gaussian approximation
            min_sample = samples[:, dim_ind].min()
            max_sample = samples[:, dim_ind].max()
            x_vals = np.linspace(min_sample, max_sample, 1000)
            y_vals = norm.pdf(x_vals, loc=mean[dim_ind], scale=std[dim_ind])
            ax_dict[dim_ind].plot(x_vals, y_vals, color="green", label="Gaussian approximation")

            # plot the true value if it is given
            if truth is not None:
                true_value = truth[dim_ind].cpu().item()
                ax_dict[dim_ind].axvline(true_value, color="red", label="true value")

            # plot the legend only for the first panel
            if dim_ind == 0:
                ax_dict[dim_ind].legend()

        plt.savefig(filename)
        plt.close(fig)
        return


# tensor object and dataloader creation (analogue to last exercise)
def createDataLoaders(
    spectra: dblArr, labels: dblArr, batch_size: int = batch_size, train_fraction: float = train_fraction, val_fraction: float = val_fraction
) -> tp.Tuple[DataLoader, DataLoader, DataLoader]:
    # Convert numpy arrays to PyTorch tensors
    spectra_tensor = torch.tensor(spectra, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.float32)

    # Split the data into training, validation, and test sets
    total_samples = len(spectra_tensor)
    train_size = int(train_fraction * total_samples)
    val_size = int(val_fraction * total_samples)
    test_size = total_samples - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(
        TensorDataset(spectra_tensor, labels_tensor), [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42)
    )

    # Create DataLoaders for batching and shuffling the data
    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        DataLoader(val_dataset, batch_size=batch_size, shuffle=False),
        DataLoader(test_dataset, batch_size=batch_size, shuffle=False),
    )


def nf_loss(inputs: torch.Tensor, batch_labels: torch.Tensor, model: nn.Module) -> torch.Tensor:
    """
    Computes the loss for a normalizing flow model.

    Parameters
    ----------
    inputs : torch.Tensor
        The input data to the model.
    batch_labels : torch.Tensor
        The labels corresponding to the input data.
    model : torch.nn.Module
        The normalizing flow model used for evaluation.
    Returns
    -------
    torch.Tensor
        The computed loss value.
    """
    log_pdfs = model.log_pdf_evaluation(batch_labels, inputs)  # pyright: ignore[reportCallIssue] # get the probability of the labels given the input data
    return -log_pdfs.mean()  # take the negative mean of the log probabilities


def evaluate_and_plot(
    model: nn.Module,
    loss_function: tp.Callable[..., torch.Tensor],
    device: torch.device,
    train_losses: list,
    val_losses: list,
    test_loader: DataLoader,
    n_labels: int,
    labelNames: list[str],
    ranges: dblArr,
    plot_folder: str = "/plots",
    suffix: str = "",
) -> None:
    """
    Evaluate the model on the test dataset and plot the results.

    Helper function to plot the training progress.

    Parameters
    ----------
    model : torch.nn.Module
        The trained model to evaluate.
    loss_function : function
        The loss function to use for evaluation.
    device : torch.device
        The device on which to evaluate the model.
    train_losses : list
        The training losses for each epoch.
    val_losses : list
        The validation losses for each epoch.
    test_loader : torch.utils.data.DataLoader
        The DataLoader object for the test dataset.
    n_labels : int
        The number of labels to evaluate.
    labelNames : list[str]
        The names of the labels.
    ranges : list
        The normalization ranges for the data.
    plot_folder : str, optional
        The folder where the plots will be saved (default is "/plots").
    suffix : str, optional
        The suffix to add to the plot filenames (default is "").
    Returns
    -------
    None
    """
    print("Evaluating and plotting model on the test dataset...")
    all_predictions, normalized_true_labels, first_batch_input, first_batch_labels = evaluate_model(model, test_loader, loss_function, device)
    # Denormalize predictions
    pred_mean = denormalize(all_predictions[:, :n_labels], ranges)
    # Extract the predicted standard deviations
    pred_std = denormalize_std(all_predictions[:, n_labels:], ranges)
    all_true_labels = denormalize(normalized_true_labels, ranges)  # Denormalize true labels

    # Check if the "plots" directory exists, if not, create it
    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)

    # plot a PDF of event 0
    b_index = 0
    model.visualize_pdf(
        first_batch_input,
        "%s/event_pdf_%s.png" % (plot_folder, suffix),
        batch_index=b_index,
        truth=first_batch_labels[b_index],  # pyright: ignore[reportCallIssue, reportOptionalSubscript]
    )

    # Plot training and validation loss
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Training Loss")
    plt.plot(range(1, len(val_losses) + 1), val_losses, label="Validation Loss")
    # plt.yscale('log')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig("%s/training_validation_loss_%s.png" % (plot_folder, suffix))

    # Scatter plots for predictions
    plt.figure(figsize=(16, 7.5))
    for j in range(n_labels):
        plt.subplot(1, 3, j + 1)
        gt = all_true_labels
        plt.scatter(gt[:, j], y=pred_mean[:, j], s=6, alpha=0.2)
        plt.plot(
            [gt[:, j].min().item(), gt[:, j].max().item()],
            [gt[:, j].min().item(), gt[:, j].max().item()],
            c="black",
            linestyle="dashed",
            label="Perfect prediction",
        )
        plt.xlabel("true " + labelNames[j])
        plt.ylabel("predicted " + labelNames[j])
        plt.legend()
    plt.tight_layout()
    plt.savefig("%s/scatter_%s.png" % (plot_folder, suffix))

    plt.figure(figsize=(16, 7.5))
    for j in range(n_labels):
        plt.subplot(1, 3, j + 1)
        gt = all_true_labels[:, j]
        pred = pred_mean[:, j]
        std = pred_std[:, j]  # Extract the predicted standard deviations
        diff = gt - pred
        plt.hist(diff, bins=50, alpha=0.75, color="skyblue", edgecolor="black")
        plt.xlabel(f"True - Predicted {labelNames[j]}")
        plt.ylabel("Frequency")
        plt.title(f"Distribution for {labelNames[j]}")
        plt.axvline(diff.mean(), color="red", linestyle="dashed", linewidth=1)
        # plt.axvline(diff.std(), color='green', linestyle='dashed', linewidth=1)
        plt.text(
            0.95,
            0.95,
            f"Mean: {diff.mean():.2f}\nStd: {diff.std():.2f}",
            transform=plt.gca().transAxes,
            fontsize=12,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(facecolor="white", alpha=0.5),
        )
    plt.tight_layout()
    plt.savefig("%s/true_predicted_%s.png" % (plot_folder, suffix))

    plt.figure(figsize=(16, 7.5))
    for j in range(n_labels):
        plt.subplot(1, 3, j + 1)
        gt = all_true_labels[:, j]
        pred = pred_mean[:, j]
        std = pred_std[:, j]  # Extract the predicted standard deviations
        plt.hist(std, bins=50, alpha=0.75, color="skyblue", edgecolor="black")
        plt.xlabel(f"STD for {labelNames[j]}")
        plt.ylabel("Frequency")
        plt.title(f"Distribution of STD for {labelNames[j]}")
        plt.axvline(std.mean(), color="red", linestyle="dashed", linewidth=1)
        # plt.axvline(std.std(), color='green', linestyle='dashed', linewidth=1)
        plt.text(
            0.95,
            0.95,
            f"Mean: {std.mean():.2f}\nStd: {std.std():.2f}",
            transform=plt.gca().transAxes,
            fontsize=12,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(facecolor="white", alpha=0.5),
        )
    plt.tight_layout()
    plt.savefig("%s/std_%s.png" % (plot_folder, suffix))
    plt.close("all")

    # Plot pull distributions for the three labels
    plt.figure(figsize=(16, 7.5))
    for j in range(n_labels):
        plt.subplot(1, 3, j + 1)
        gt = all_true_labels[:, j]
        pred = pred_mean[:, j]
        std = pred_std[:, j]  # Extract the predicted standard deviations
        pull = (gt - pred) / std  # Calculate the pull
        plt.hist(pull, bins=50, alpha=0.75, color="skyblue", edgecolor="black")
        plt.xlabel(f"Pull for {labelNames[j]}")
        plt.ylabel("Frequency")
        plt.title(f"Pull Distribution for {labelNames[j]}")
        plt.axvline(pull.mean(), color="red", linestyle="dashed", linewidth=1)
        # plt.axvline(pull.std(), color='green', linestyle='dashed', linewidth=1)
        plt.text(
            0.95,
            0.95,
            f"Mean: {pull.mean():.2f}\nStd: {pull.std():.2f}",
            transform=plt.gca().transAxes,
            fontsize=12,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(facecolor="white", alpha=0.5),
        )
    plt.tight_layout()
    plt.savefig("%s/pull_%s.png" % (plot_folder, suffix))

    return


def main() -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument("--normalizing_flow_type", default="diagonal_gaussian", choices=["diagonal_gaussian", "full_gaussian", "full_flow"])
    args = parser.parse_args()
    print("Using normalizing flow type ", args.normalizing_flow_type)

    # Detect and use Apple Silicon GPU (MPS) if available
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    # MPS does not support double precision, therefore we need to run the flow on the CPU
    fp64_on_cpu = True if args.normalizing_flow_type == "full_flow" and device.type == "mps" else False
    print(f"Using device: {device}, performing fp64 on CPU: {fp64_on_cpu}")

    model = CombinedModel(TinyCNNEncoder, fp64_on_cpu=fp64_on_cpu, nf_type=args.normalizing_flow_type)
    model.to(device)

    # Call the function to get normalized data
    spectra, labels, spectra_length, n_labels, labelNames, ranges = get_normalized_data(DATA_PATH)  # pyright: ignore[reportAssignmentType]
    train_loader, val_loader, test_loader = createDataLoaders(spectra, labels)

    plot_dir = "plots/%s" % args.normalizing_flow_type
    loss_function = nf_loss
    plot_function = partial(
        evaluate_and_plot,
        test_loader=test_loader,
        n_labels=n_labels,
        labelNames=labelNames,
        ranges=ranges,
        plot_folder=plot_dir,
    )
    model_name = "NN_1_flow_%s" % args.normalizing_flow_type

    # train the model
    train_losses, val_losses, best_model = train_model(
        model,
        train_loader,
        val_loader,
        loss_function,
        learning_rate,
        num_epochs,
        patience,
        device,
        plot_fn=plot_function,
        plot_interval=10,
        model_name=model_name,
    )

    model.load_state_dict(best_model)  # pyright: ignore[reportArgumentType] # Load the best model, the last model might not be the best
    model.to(device)

    # plot final model
    evaluate_and_plot(
        model,
        loss_function,
        device,
        train_losses,
        val_losses,
        test_loader,
        n_labels,
        labelNames,
        ranges,
        plot_folder=plot_dir,
        suffix="final",
    )

    # convert all training+final pngs into an animated gif
    png_files = glob.glob(os.path.join(plot_dir, "*.png"))

    # Extract unique prefixes (assumes filenames are like 'name1_000010.png')
    final_png_files = [f[:-10] for f in glob.glob(os.path.join(plot_dir, "*final.png"))]
    prefixes = [f for f in final_png_files]

    # Loop through each prefix and generate a GIF
    # needs "imagemagick" installed
    for prefix in prefixes:
        # Build the command string
        command = f"convert -delay 20 -loop 0 {prefix}_*.png {prefix}.gif"
        # Optional: print the command for debugging
        print(f"Executing: {command}")
        # Execute the command
        subprocess.run(command, shell=True, check=True)

    return os.EX_OK


if __name__ == "__main__":
    sys.exit(main())
