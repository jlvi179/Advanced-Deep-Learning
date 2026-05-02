import torch
import torch.nn as nn


class TinyCNNWithUncertainty(nn.Module):
    """
    A CNN model that outputs both predictions and their uncertainties (standard deviations).
    
    The model outputs 2*n_labels values: n_labels for the mean predictions and n_labels for log-std predictions.
    The log-std is used instead of std to ensure positive standard deviations.
    """
    def __init__(self, n_labels):
        super(TinyCNNWithUncertainty, self).__init__()
        
        # Shared feature extraction layers
        self.features = nn.Sequential(
            nn.Conv1d(1, 10, kernel_size=5),
            nn.ReLU(),
            nn.BatchNorm1d(10),
            nn.Dropout(0.1),
            nn.AvgPool1d(3),

            nn.Conv1d(10, 20, kernel_size=5),
            nn.ReLU(),
            nn.BatchNorm1d(20),
            nn.Dropout(0.1),
            nn.AvgPool1d(3),

            nn.Conv1d(20, 40, kernel_size=5),
            nn.ReLU(),
            nn.BatchNorm1d(40),
            nn.Dropout(0.1),
            nn.AvgPool1d(3),

            nn.Conv1d(40, 10, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(10),
            nn.Dropout(0.1),
            nn.AvgPool1d(2),

            nn.Conv1d(10, 12, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm1d(12),
            nn.Dropout(0.2),

            nn.Conv1d(12, 10, kernel_size=1),
            nn.Dropout(0.2),
        )
        
        # Fully connected layers after feature extraction
        self.fc_shared = nn.Sequential(
            nn.Linear(300, 32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(10 * 32, 128),
            nn.ReLU(),
        )
        
        # Output heads: one for means, one for log-stds
        self.fc_mean = nn.Linear(128, n_labels)
        self.fc_logstd = nn.Linear(128, n_labels)
        
        self.n_labels = n_labels

    def forward(self, x):
        """
        Forward pass that returns both predictions and their log-standard deviations.
        
        Parameters
        ----------
        x : torch.Tensor
            Input spectra of shape (batch_size, 1, spectra_length)
            
        Returns
        -------
        torch.Tensor
            Concatenated output of shape (batch_size, 2*n_labels) where:
            - First n_labels values are the mean predictions
            - Last n_labels values are log-std predictions
        """
        x = self.features(x)
        x = self.fc_shared(x)
        
        mean = self.fc_mean(x)
        logstd = self.fc_logstd(x)
        
        # Concatenate means and log-stds
        output = torch.cat([mean, logstd], dim=1)
        return output
    
    def get_predictions_and_uncertainties(self, x):
        """
        Forward pass that returns predictions and uncertainties separately.
        
        Parameters
        ----------
        x : torch.Tensor
            Input spectra
            
        Returns
        -------
        tuple
            (predictions, std_predictions) where std_predictions are the standard deviations
        """
        output = self.forward(x)
        mean = output[:, :self.n_labels]
        logstd = output[:, self.n_labels:]
        std = torch.exp(logstd)
        return mean, std
