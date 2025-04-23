import os
import torch
import random
import numpy as np
import datetime
import matplotlib.pyplot as plt

def set_seed(seed_value):
    """Sets the seed for reproducibility."""
    if seed_value != -1:
        random.seed(seed_value)
        np.random.seed(seed_value)
        torch.manual_seed(seed_value)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print(f"Seeds set to {seed_value}")
    else:
        print("Seeds not set (random initialization)")

def get_device():
    """Gets the appropriate device (CUDA or CPU)."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device available: {device}')
    return device

def get_timestamp():
    """Returns the current timestamp as a string."""
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def save_plot(fig, directory, filename):
    """Saves a matplotlib figure to the specified directory."""
    os.makedirs(directory, exist_ok=True)
    filepath = os.path.join(directory, filename)
    fig.savefig(filepath)
    print(f"Plot saved to: {filepath}")
    plt.close(fig) # Close the figure after saving to free memory