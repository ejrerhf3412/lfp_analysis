# src/signal_processing.py
import numpy as np
from scipy.signal import butter, filtfilt, hilbert

def calculate_amplitude_envelopes_and_filtered_lfp(lfp_data, freq_bands, fs, filter_order):
    """
    Calculates amplitude envelopes for each band and stores filtered LFP.
    Returns two dictionaries:
        band_envelopes_all_channels: {band_name: envelope_array}
        filtered_lfp_for_phase_calc: {band_name: filtered_lfp_array}
    """
    band_envelopes_all_channels = {}
    filtered_lfp_storage = {}
    print("Pre-calculating amplitude envelopes and storing filtered LFP for phase...")

    for band_name, (low_freq, high_freq) in freq_bands.items():
        # print(f"  Processing band: {band_name} for amplitude and storing filtered LFP...")
        nyquist_freq = 0.5 * fs
        low = low_freq / nyquist_freq
        high = high_freq / nyquist_freq
        if high >= 1.0: high = 0.999
        if low <= 0: low = 1e-5
        
        b, a = butter(filter_order, [low, high], btype='band')
        _filtered_lfp = filtfilt(b, a, lfp_data, axis=1)
        
        filtered_lfp_storage[band_name] = _filtered_lfp # Store for potential on-demand phase
        
        analytic_signal_band = hilbert(_filtered_lfp, axis=1)
        band_envelopes_all_channels[band_name] = np.abs(analytic_signal_band)
    print("Amplitude envelopes and filtered LFP storage done.")
    return band_envelopes_all_channels, filtered_lfp_storage

def precompute_all_phase_series(lfp_data, filtered_lfp_storage, freq_bands, fs):
    """
    Precomputes full phase time series for raw LFP and all specified filtered bands.
    Returns:
        all_phase_series_cache: {(is_filtered, band_name_or_None): phase_array_400xtp}
    """
    all_phase_series_cache = {}
    print("Pre-calculating ALL phase time series (Raw and Filtered)...")

    # Raw LFP phase
    analytic_signal_raw_lfp = hilbert(lfp_data, axis=1)
    all_phase_series_cache[(False, None)] = np.angle(analytic_signal_raw_lfp)
    # print("  Raw LFP phase done.")

    # Filtered LFP phases
    for band_name in freq_bands.keys():
        # print(f"  Calculating phase for pre-stored filtered band: {band_name} (entire series)")
        _filtered_lfp = filtered_lfp_storage[band_name] # Use pre-filtered LFP
        analytic_signal_band_filt = hilbert(_filtered_lfp, axis=1)
        all_phase_series_cache[(True, band_name)] = np.angle(analytic_signal_band_filt)
    print("All phase time series precomputation finished.")
    return all_phase_series_cache

def correct_phase_diff(diff): # Also used by singularity detection
    """Ensures phase difference is in [-pi, pi]."""
    return (diff + np.pi) % (2 * np.pi) - np.pi