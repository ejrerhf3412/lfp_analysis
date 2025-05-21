# src/singularity_detection.py
import numpy as np
from scipy.ndimage import zoom
from .signal_processing import correct_phase_diff # Assuming correct_phase_diff is here or in utils

def detect_singularities(phase_grid_zoomed, phase_tolerance):
    """Detects singularities on a (potentially zoomed) phase grid."""
    r_max, c_max = phase_grid_zoomed.shape
    s_list, a_list = [], []
    for r_idx in range(r_max - 1):
        for c_idx in range(c_max - 1):
            p1 = phase_grid_zoomed[r_idx, c_idx]
            p2 = phase_grid_zoomed[r_idx, c_idx + 1]
            p3 = phase_grid_zoomed[r_idx + 1, c_idx + 1]
            p4 = phase_grid_zoomed[r_idx + 1, c_idx]
            
            diffs = [correct_phase_diff(p2 - p1),
                     correct_phase_diff(p3 - p2),
                     correct_phase_diff(p4 - p3),
                     correct_phase_diff(p1 - p4)]
            total_phase_change = sum(diffs)
            
            if np.abs(total_phase_change - 2 * np.pi) < phase_tolerance:
                s_list.append((r_idx, c_idx)) # Store as (row, col) in the given grid's coordinate
            elif np.abs(total_phase_change + 2 * np.pi) < phase_tolerance:
                a_list.append((r_idx, c_idx))
    return s_list, a_list

def get_zoomed_phase_grid_from_series(time_idx, 
                                      full_phase_series_400xtp, 
                                      grid_dim, 
                                      upsample_factor, 
                                      interpolation_order):
    """
    Extracts phase for a time_idx from a full series, reshapes, and zooms.
    """
    phase_data_flat = full_phase_series_400xtp[:, time_idx]
    phase_grid_original_20x20 = phase_data_flat.reshape((grid_dim, grid_dim))
    
    # Interpolate complex representation for phase
    cos_phi = np.cos(phase_grid_original_20x20)
    sin_phi = np.sin(phase_grid_original_20x20)
    zoomed_cos_phi = zoom(cos_phi, upsample_factor, order=interpolation_order)
    zoomed_sin_phi = zoom(sin_phi, upsample_factor, order=interpolation_order)
    return np.arctan2(zoomed_sin_phi, zoomed_cos_phi)