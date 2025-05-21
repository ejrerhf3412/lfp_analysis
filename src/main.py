# src/main.py
import time
from . import config # Imports all variables from config.py
from . import data_loader
from . import signal_processing
from . import track_management # This now handles its own singularity_detection import if needed
from . import visualization

def run_analysis_and_gui():
    # 1. Load Data
    lfp_data, num_channels, num_time_points = data_loader.load_lfp_data(
                                                    config.MAT_FILE_PATH, 
                                                    config.VARIABLE_NAME
                                                )
    data_loader.verify_grid_compatibility(num_channels, config.GRID_DIM)

    # 2. Precompute Signal Properties (Amplitude Envelopes and ALL Phase Series)
    amp_env_data, filtered_lfp_data = \
        signal_processing.calculate_amplitude_envelopes_and_filtered_lfp(
            lfp_data, config.FREQ_BANDS, config.FS, config.FILTER_ORDER
        )
    
    all_phase_series_cache_data = \
        signal_processing.precompute_all_phase_series(
            lfp_data, filtered_lfp_data, config.FREQ_BANDS, config.FS
        )

    # 3. Precompute All Tracks for ALL conditions
    # This function will populate track_management.all_tracks_cache
    all_tracks_cache_data = track_management.perform_all_track_precomputations(
        all_phase_series_cache_data, 
        config.FREQ_BANDS,
        config.GRID_DIM, 
        config.UPSAMPLE_FACTOR, 
        config.INTERPOLATION_ORDER_ZOOM,
        config.MAX_TRACK_DISTANCE_SQ,
        config.PHASE_TOLERANCE,
        config.MAX_POINTS_PER_TRACK_DEQUE,
        num_time_points # Pass num_time_points here
    )

    # 4. Launch Visualization GUI
    visualization.launch_gui(
        amp_env_data,
        all_phase_series_cache_data, # For instantaneous singularity display
        all_tracks_cache_data,       # For displaying precomputed tracks
        config,                      # Pass the config module for GUI to access constants
        num_time_points
    )

if __name__ == "__main__":
    run_analysis_and_gui()