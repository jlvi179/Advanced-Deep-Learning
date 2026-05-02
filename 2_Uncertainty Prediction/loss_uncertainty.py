import torch
import torch.nn as nn
import math


class GaussianNLLLoss(nn.Module):
    """
    Negative Log-Likelihood loss for Gaussian distributions.
    
    This loss function assumes the model outputs both the mean and log-standard deviation
    for each prediction. It computes the negative log-likelihood of a Gaussian distribution.
    
    The NLL for a Gaussian is:
    NLL = 0.5 * log(2*pi*sigma^2) + 0.5 * ((y - mu) / sigma)^2
        = 0.5 * log(2*pi) + log(sigma) + 0.5 * ((y - mu) / sigma)^2
    """
    
    def __init__(self, n_labels=3):
        super(GaussianNLLLoss, self).__init__()
        self.n_labels = n_labels
        
    def forward(self, model_output, targets):
        """
        Compute the Gaussian NLL loss.
        
        Parameters
        ----------
        model_output : torch.Tensor
            Model output of shape (batch_size, 2*n_labels) where:
            - First n_labels columns are the predicted means
            - Last n_labels columns are the predicted log-stds
        targets : torch.Tensor
            Target labels of shape (batch_size, n_labels)
            
        Returns
        -------
        torch.Tensor
            Scalar loss value (mean NLL over all samples and dimensions)
        """
        # Extract means and log-stds from model output
        means = model_output[:, :self.n_labels]
        log_stds = model_output[:, self.n_labels:]
        
        # Compute stds from log-stds (exp to ensure positive values)
        stds = torch.exp(log_stds)
        
        # Compute the Gaussian NLL
        # NLL = 0.5 * log(2*pi*sigma^2) + 0.5 * ((y - mu) / sigma)^2
        nll = 0.5 * torch.log(2 * math.pi * stds**2) + 0.5 * ((targets - means) / stds)**2
        
        # Return mean loss over batch and all dimensions
        return nll.mean()


def nll_loss(predictions, batch_labels, n_labels):
    """
    Slide-style negative log-likelihood loss.

    Implements the simple NLL variant from the lecture slides:
    - predictions: first `n_labels` are means, next `n_labels` are log-stds
    - returns mean over batch and labels of 0.5*((y-mu)/sigma)**2 + log_sigma
    """
    # extract the mean values
    mean = predictions[:, :n_labels]
    # extract the log-standard deviation values
    log_std = predictions[:, n_labels:]
    # convert log standard deviation to standard deviation
    std = torch.exp(log_std)
    # NLL formula (constant terms omitted as in slides)
    return torch.mean(0.5 * ((batch_labels - mean) / std) ** 2 + log_std)


class GaussianNLLLossCallable:
    """
    Callable wrapper for Gaussian NLL loss that works with the training loop.
    
    This class allows the loss to be called in the format: loss_fn(spectra, labels, model)
    which is compatible with the existing training infrastructure.
    """
    
    def __init__(self, n_labels=3):
        self.nll_loss = GaussianNLLLoss(n_labels)
        self.n_labels = n_labels
        
    def __call__(self, spectra, labels, model):
        """
        Compute loss given spectra, labels, and model.
        
        Parameters
        ----------
        spectra : torch.Tensor
            Input spectra (batch_size, 1, spectra_length)
        labels : torch.Tensor
            Target labels (batch_size, n_labels)
        model : torch.nn.Module
            The neural network model
            
        Returns
        -------
        torch.Tensor
            Scalar loss value
        """
        model_output = model(spectra)
        return self.nll_loss(model_output, labels)
