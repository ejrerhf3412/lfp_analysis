# src/data_loader.py
import scipy.io
import numpy as np

def load_lfp_data(filepath, variable_name):
    """Loads LFP data from a .mat file."""
    try:
        mat_contents = scipy.io.loadmat(filepath)
        lfp_data_all_channels = mat_contents[variable_name]
        print(f"Data loaded successfully. Shape: {lfp_data_all_channels.shape}")
        num_total_channels, num_time_points = lfp_data_all_channels.shape
        return lfp_data_all_channels, num_total_channels, num_time_points
    except FileNotFoundError:
        print(f"FNF Error: File '{filepath}' not found.")
        exit()
    except KeyError:
        print(f"KE Error: Variable '{variable_name}' not found in '{filepath}'.")
        exit()
    except Exception as e:
        print(f"An unexpected error occurred during data loading: {e}")
        exit()

def verify_grid_compatibility(num_total_channels, grid_dim):
    """Verifies if the number of channels matches the grid dimensions."""
    if num_total_channels != grid_dim * grid_dim:
        print(f"Grid Error: Number of channels ({num_total_channels}) "
              f"does not match GRID_DIM^2 ({grid_dim*grid_dim}).")
        exit()