# src/config.py

# --- File and Data Configuration ---
MAT_FILE_PATH = '/Users/xxx/Desktop/extracted_lfp_segments_fixed.mat' # LFP data recorded from the electrode array
VARIABLE_NAME = 'lfp_data_top400'
FS = 1000 # Sampling rate in Hz

# --- Signal Processing Configuration ---
FREQ_BANDS = {
    'Delta (δ)': (0.5, 4),
    'Theta (θ)': (4, 8),
    'Alpha (α)': (8, 13),
    'Sigma (σ)': (12, 16),
    'Beta (β)': (13, 30),
    'Gamma (γ)': (30, 80)
}
FILTER_ORDER = 4

# --- Visualization and Grid Configuration ---
AMPLITUDE_CMAP = 'viridis'
GRID_DIM = 20
UPSAMPLE_FACTOR = 3
INTERPOLATION_ORDER_ZOOM = 1 # 1=bilinear for zoom

# --- Singularity Detection Configuration ---
PHASE_TOLERANCE = np.pi / 2 # Tolerance for sum being close to +/- 2*pi

# --- Track Management Configuration ---
MAX_PIXEL_DISPLACEMENT_FOR_TRACK_MATCHING_ZOOMED = 7.0 # Max pixel distance in ZOOMED grid
MAX_TRACK_DISTANCE_SQ = MAX_PIXEL_DISPLACEMENT_FOR_TRACK_MATCHING_ZOOMED**2
TRACK_LINEWIDTH = 1.5
MAX_POINTS_PER_TRACK_DEQUE = 50 # Max points in deque for display of active tracks

# --- GUI Initial States ---
INITIAL_TIME_IDX = 0
# INITIAL_BAND_NAME will be the first key from FREQ_BANDS
INITIAL_SHOW_SINGULARITIES = True
INITIAL_FILTER_PHASE_FOR_SINGULARITIES = False
INITIAL_SHOW_TRACKS = False # Start with tracks OFF for potentially faster first load