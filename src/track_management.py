# src/track_management.py
import numpy as np
from collections import deque
import uuid
import time # For timing precomputation
from .singularity_detection import detect_singularities, get_zoomed_phase_grid_from_series

all_tracks_cache = {} # Global cache for fully computed track lists for different conditions

def precompute_all_tracks_for_condition(full_phase_time_series, 
                                        condition_key_tuple,
                                        num_time_points, # Added num_time_points
                                        grid_dim, upsample_factor, interpolation_order, # for get_zoomed_phase_grid
                                        max_track_distance_sq, phase_tolerance, # for detection & matching
                                        max_points_per_track_deque): # for deque
    """
    Precomputes all track segments using a given full phase time series.
    """
    global all_tracks_cache # Allow modification of the global cache
    
    # Check if already computed (should ideally not happen if called once per condition from main)
    # if condition_key_tuple in all_tracks_cache:
    #     print(f"  Tracks already in cache for {condition_key_tuple}. Returning cached.")
    #     return all_tracks_cache[condition_key_tuple]

    print(f"\n--- Precomputing All Tracks for Condition: {condition_key_tuple} ---")
    start_comp_time = time.time()
    
    condition_specific_tracks = []
    current_active_spirals = [] 
    current_active_anti = []    

    for t in range(num_time_points):
        if t % 500 == 0 and t > 0: 
            print(f"    Precomputing tracks at frame {t}/{num_time_points} for {condition_key_tuple}...")
        
        phase_grid_zoomed = get_zoomed_phase_grid_from_series(
                                t, full_phase_time_series, grid_dim, 
                                upsample_factor, interpolation_order
                            )
        s_t, a_t = detect_singularities(phase_grid_zoomed, phase_tolerance)

        # Manage Spiral Tracks
        newly_active_s = []; s_t_matched_status = [False] * len(s_t)
        for track_dict in current_active_spirals:
            last_r_z, last_c_z = track_dict['points_rc_zoomed'][-1]
            best_match_idx = -1; min_dist_sq = float('inf')
            for i, (curr_r_z, curr_c_z) in enumerate(s_t):
                if not s_t_matched_status[i]:
                    dist_sq = (curr_r_z - last_r_z)**2 + (curr_c_z - last_c_z)**2
                    if dist_sq <= max_track_distance_sq:
                        if dist_sq < min_dist_sq: min_dist_sq = dist_sq; best_match_idx = i
            if best_match_idx != -1:
                track_dict['points_rc_zoomed'].append(s_t[best_match_idx])
                track_dict['time_indices'].append(t); track_dict['end_time'] = t
                newly_active_s.append(track_dict); s_t_matched_status[best_match_idx] = True
            else: condition_specific_tracks.append(track_dict) 
        for i, pt_s in enumerate(s_t):
            if not s_t_matched_status[i]:
                newly_active_s.append({'id':uuid.uuid4(), 'type':'spiral', 
                                     'points_rc_zoomed':deque([pt_s], maxlen=max_points_per_track_deque), 
                                     'time_indices':deque([t], maxlen=max_points_per_track_deque),
                                     'start_time':t, 'end_time':t})
        current_active_spirals = newly_active_s

        # Manage Anti-Spiral Tracks
        newly_active_as = []; a_t_matched_status = [False] * len(a_t)
        for track_dict in current_active_anti:
            last_r_z, last_c_z = track_dict['points_rc_zoomed'][-1]
            best_match_idx = -1; min_dist_sq = float('inf')
            for i, (curr_r_z, curr_c_z) in enumerate(a_t):
                if not a_t_matched_status[i]:
                    dist_sq = (curr_r_z - last_r_z)**2 + (curr_c_z - last_c_z)**2
                    if dist_sq <= max_track_distance_sq:
                        if dist_sq < min_dist_sq: min_dist_sq = dist_sq; best_match_idx = i
            if best_match_idx != -1:
                track_dict['points_rc_zoomed'].append(a_t[best_match_idx])
                track_dict['time_indices'].append(t); track_dict['end_time'] = t
                newly_active_as.append(track_dict); a_t_matched_status[best_match_idx] = True
            else: condition_specific_tracks.append(track_dict)
        for i, pt_a in enumerate(a_t):
            if not a_t_matched_status[i]:
                newly_active_as.append({'id':uuid.uuid4(), 'type':'anti_spiral', 
                                      'points_rc_zoomed':deque([pt_a], maxlen=max_points_per_track_deque), 
                                      'time_indices':deque([t], maxlen=max_points_per_track_deque),
                                      'start_time':t, 'end_time':t})
        current_active_anti = newly_active_as
    
    condition_specific_tracks.extend(current_active_spirals)
    condition_specific_tracks.extend(current_active_anti)
    
    all_tracks_cache[condition_key_tuple] = condition_specific_tracks
    end_comp_time = time.time()
    tdiff = end_comp_time - start_comp_time
    print(f"--- Finished Precomputing Tracks for {condition_key_tuple}. Found {len(condition_specific_tracks)} segments. Took {tdiff:.2f}s ---")
    return condition_specific_tracks

def perform_all_track_precomputations(all_phase_series_cache_data, freq_bands_config, 
                                      grid_dim_config, upsample_factor_config, 
                                      interpolation_order_config, max_track_distance_sq_config,
                                      phase_tolerance_config, max_points_per_track_deque_config,
                                      num_time_points_data): # Added num_time_points
    """Main loop to call precompute_all_tracks_for_condition for all conditions."""
    print("\n===== STARTING ALL TRACK PRECOMPUTATIONS (via track_management) =====")
    start_total_precomp_time = time.time()
    
    # 1. For Raw LFP phase
    raw_phase_key = (False, None)
    if raw_phase_key in all_phase_series_cache_data:
        precompute_all_tracks_for_condition(
            all_phase_series_cache_data[raw_phase_key], raw_phase_key,
            num_time_points_data, grid_dim_config, upsample_factor_config,
            interpolation_order_config, max_track_distance_sq_config,
            phase_tolerance_config, max_points_per_track_deque_config
        )
    else:
        print(f"Error: Raw phase series not found in cache for key {raw_phase_key}")

    # 2. For each filtered band's phase
    for band_name_iter in freq_bands_config.keys():
        filtered_phase_key = (True, band_name_iter)
        if filtered_phase_key in all_phase_series_cache_data:
            precompute_all_tracks_for_condition(
                all_phase_series_cache_data[filtered_phase_key], filtered_phase_key,
                num_time_points_data, grid_dim_config, upsample_factor_config,
                interpolation_order_config, max_track_distance_sq_config,
                phase_tolerance_config, max_points_per_track_deque_config
            )
        else:
            print(f"Error: Filtered phase series not found in cache for key {filtered_phase_key}")
            
    end_total_precomp_time = time.time()
    tdiff_total = end_total_precomp_time - start_total_precomp_time
    print(f"===== ALL TRACK PRECOMPUTATIONS FINISHED in {tdiff_total:.2f}s (via track_management) =====")
    return all_tracks_cache # Return the populated cache