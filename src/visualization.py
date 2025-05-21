# src/visualization.py
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons, CheckButtons
import numpy as np
from scipy.ndimage import zoom # For amplitude display
from .singularity_detection import detect_singularities, get_zoomed_phase_grid_from_series # For instantaneous sings

# --- Global variables for GUI state and Matplotlib objects (can be refactored into a class) ---
fig, ax = None, None
img_display = None
spiral_scatter, anti_spiral_scatter = None, None
legend_singularities = None
cbar = None

time_slider, band_radio = None, None
check_singularities, check_filter_phase, check_show_tracks = None, None, None

# Data holders (will be passed from main)
config_module = None # To hold the config object/module
band_envelopes_data = None
all_phase_series_data_cache = None # The full cache of raw and filtered phase series
all_tracks_data_cache = None       # The full cache of precomputed tracks
num_time_points_data = 0

# View state
current_selected_band_view = ""
current_display_time_idx_view = 0
current_show_singularities_view = False
current_filter_phase_view = False # For instantaneous singularity phase source
current_show_tracks_view = False
current_tracks_for_display_plot = [] # Tracks for the current (filter_phase, band) condition
track_lines_plots_display = []       # Matplotlib Line2D objects for tracks

def launch_gui(_band_envelopes, _all_phase_series, _all_tracks, _config, _num_time_points):
    """Initializes and shows the Matplotlib GUI."""
    global fig, ax, img_display, spiral_scatter, anti_spiral_scatter, legend_singularities, cbar
    global time_slider, band_radio, check_singularities, check_filter_phase, check_show_tracks
    global config_module, band_envelopes_data, all_phase_series_data_cache, all_tracks_data_cache, num_time_points_data
    global current_selected_band_view, current_display_time_idx_view, \
           current_show_singularities_view, current_filter_phase_view, \
           current_show_tracks_view, current_tracks_for_display_plot, track_lines_plots_display


    # Store passed data
    config_module = _config
    band_envelopes_data = _band_envelopes
    all_phase_series_data_cache = _all_phase_series
    all_tracks_data_cache = _all_tracks
    num_time_points_data = _num_time_points
    track_lines_plots_display = []


    fig, ax = plt.subplots(figsize=(10, 7))
    plt.subplots_adjust(left=0.30, bottom=0.25, right=0.90, top=0.9)

    # Initial state values for widgets and logic
    current_display_time_idx_view = config_module.INITIAL_TIME_IDX
    current_selected_band_view = list(config_module.FREQ_BANDS.keys())[0]
    current_show_singularities_view = config_module.INITIAL_SHOW_SINGULARITIES
    current_filter_phase_view = config_module.INITIAL_FILTER_PHASE_FOR_SINGULARITIES # For instantaneous sings
    current_show_tracks_view = config_module.INITIAL_SHOW_TRACKS

    # Initial plot setup
    initial_envelope_data = band_envelopes_data[current_selected_band_view][:, current_display_time_idx_view]
    initial_env_grid_orig = initial_envelope_data.reshape((config_module.GRID_DIM, config_module.GRID_DIM))
    initial_zoomed_env = zoom(initial_env_grid_orig, config_module.UPSAMPLE_FACTOR, order=config_module.INTERPOLATION_ORDER_ZOOM)
    
    img_display = ax.imshow(initial_zoomed_env, cmap=config_module.AMPLITUDE_CMAP, vmin=0, interpolation='bilinear')
    ax.set_title(f'Amplitude: {current_selected_band_view} at Time: {current_display_time_idx_view}')
    ax.set_xticks([]); ax.set_yticks([])
    cbar = fig.colorbar(img_display, ax=ax, fraction=0.046, pad=0.04); cbar.set_label('Amp. Env.')
    
    spiral_scatter = ax.scatter([],[],s=config_module.SINGULARITY_MARKER_SIZE,facecolors='none',edgecolors=config_module.SINGULARITY_EDGE_COLOR,lw=1.5,label='Spirals', zorder=10)
    anti_spiral_scatter = ax.scatter([],[],s=config_module.SINGULARITY_MARKER_SIZE,marker='x',c=config_module.SINGULARITY_ANTI_EDGE_COLOR,lw=1.5,label='Anti-spirals', zorder=10)
    legend_singularities = ax.legend(handles=[spiral_scatter,anti_spiral_scatter],loc='upper left',fontsize='small',facecolor='gray',framealpha=0.5)
    legend_singularities.set_visible(current_show_singularities_view)

    # Controls
    ax_slider_time = plt.axes([0.30, 0.1, 0.55, 0.03]); 
    time_slider = Slider(ax_slider_time, 'Time', 0, num_time_points_data-1, valinit=current_display_time_idx_view, valstep=1)
    
    try: active_band_idx = list(config_module.FREQ_BANDS.keys()).index(current_selected_band_view)
    except ValueError: active_band_idx = 0
    ax_radio_band = plt.axes([0.05,0.55,0.20,0.30]); 
    band_radio = RadioButtons(ax_radio_band, list(config_module.FREQ_BANDS.keys()), active=active_band_idx)
    
    ax_check_singularities = plt.axes([0.05,0.45,0.20,0.08]); 
    check_singularities = CheckButtons(ax_check_singularities, ['Show Singularities'], [current_show_singularities_view])
    
    ax_check_filter_phase = plt.axes([0.05,0.35,0.20,0.08]); 
    check_filter_phase = CheckButtons(ax_check_filter_phase, ['Filter Phase Signal'], [current_filter_phase_view]) # For instantaneous sings
    
    ax_check_show_tracks = plt.axes([0.05,0.25,0.20,0.08]); 
    check_show_tracks = CheckButtons(ax_check_show_tracks, ['Show Tracks'], [current_show_tracks_view])
    print("Plot and controls initialized by visualization module.")

    # Connect Callbacks
    time_slider.on_changed(lambda val: update_plot_gui("time_slider"))
    band_radio.on_clicked(lambda label: update_plot_gui("band_radio"))
    check_singularities.on_clicked(lambda label: update_plot_gui("check_singularities"))
    check_filter_phase.on_clicked(lambda label: update_plot_gui("check_filter_phase")) # Affects instantaneous sings AND track set choice
    check_show_tracks.on_clicked(lambda label: update_plot_gui("check_show_tracks"))
    print("Widget callbacks connected by visualization module.")

    # Initial track set loading
    initial_track_cond_key = (current_filter_phase_view, 
                              current_selected_band_view if current_filter_phase_view else None)
    if initial_track_cond_key in all_tracks_data_cache:
        current_tracks_for_display_plot = all_tracks_data_cache[initial_track_cond_key]
    else:
        print(f"CRITICAL ERROR in GUI: Initial track condition {initial_track_cond_key} not found in cache!")
        current_tracks_for_display_plot = []


    update_plot_gui("initial_call") # Draw initial plot
    print("Initial plot drawn by visualization module.")
    plt.show()


def update_plot_gui(event_source=None):
    """Main GUI update function, called by widget events."""
    global current_selected_band_view, current_display_time_idx_view, \
           current_show_singularities_view, current_filter_phase_view, \
           current_show_tracks_view, track_lines_plots_display, current_tracks_for_display_plot
    global config_module, band_envelopes_data, all_phase_series_data_cache, all_tracks_data_cache # Read-only access to data

    # Get current widget states
    current_display_time_idx_view = int(time_slider.val)
    current_selected_band_view = band_radio.value_selected
    current_show_singularities_view = check_singularities.get_status()[0]
    new_filter_phase_state_for_view = check_filter_phase.get_status()[0] # This is for BOTH inst. sings and track set
    current_show_tracks_view = check_show_tracks.get_status()[0]
    
    # --- Load the correct set of precomputed tracks if filter/band condition changed ---
    track_condition_key_for_display = (new_filter_phase_state_for_view, 
                                       current_selected_band_view if new_filter_phase_state_for_view else None)
    
    # Check if the condition for displaying tracks has changed
    # We need to compare with the *previous* filter_phase_view for tracks
    previous_track_condition_filter_state = current_filter_phase_view 
    # (assuming current_filter_phase_view hasn't been updated yet for *this call* if event was not check_filter_phase)
    # A bit tricky, let's just re-fetch based on new_filter_phase_state_for_view always
    
    if track_condition_key_for_display in all_tracks_data_cache:
        current_tracks_for_display_plot = all_tracks_data_cache[track_condition_key_for_display]
    else:
        print(f"Error in GUI: Tracks for condition {track_condition_key_for_display} not found in cache!")
        current_tracks_for_display_plot = []
    
    current_filter_phase_view = new_filter_phase_state_for_view # Now update the global view state

    # --- Update Amplitude Envelope ---
    envelope_data = band_envelopes_data[current_selected_band_view][:, current_display_time_idx_view]
    env_grid_orig = envelope_data.reshape((config_module.GRID_DIM, config_module.GRID_DIM))
    zoomed_amp_data = zoom(env_grid_orig, config_module.UPSAMPLE_FACTOR, order=config_module.INTERPOLATION_ORDER_ZOOM)
    img_display.set_data(zoomed_amp_data)
    min_amp, max_amp = np.min(zoomed_amp_data), np.max(zoomed_amp_data)
    if max_amp > min_amp: img_display.set_clim(min_amp, max_amp)
    else: # Handle flat data
        val=max_amp
        if np.isclose(val,0): img_display.set_clim(-0.05,0.05)
        else: 
            abs_v=np.abs(val)
            clim_min=val - abs_v*0.05 if val!=0 else -0.01
            clim_max=val + abs_v*0.05 if val!=0 else 0.01
            if np.isclose(clim_min,clim_max) and not np.isclose(clim_min,-0.05): clim_min-=0.01; clim_max+=0.01
            img_display.set_clim(clim_min,clim_max)
    ax.set_title(f'Amp: {current_selected_band_view} @T: {current_display_time_idx_view}')

    # --- Update Singularity Markers (Instantaneous) ---
    if current_show_singularities_view:
        phase_series_for_inst_sings = all_phase_series_data_cache[(current_filter_phase_view, 
                                                                  current_selected_band_view if current_filter_phase_view else None)]
        phase_grid_for_inst_sings = get_zoomed_phase_grid_from_series(
                                            current_display_time_idx_view, 
                                            phase_series_for_inst_sings,
                                            config_module.GRID_DIM,
                                            config_module.UPSAMPLE_FACTOR,
                                            config_module.INTERPOLATION_ORDER_ZOOM
                                        )
        if phase_grid_for_inst_sings is not None:
            s_disp, a_disp = detect_singularities(phase_grid_for_inst_sings, config_module.PHASE_TOLERANCE)
            if s_disp: sr,sc=zip(*s_disp); spiral_scatter.set_offsets(np.c_[[c+0.5 for c in sc],[r+0.5 for r in sr]])
            else: spiral_scatter.set_offsets(np.empty((0,2)))
            if a_disp: asr,asc=zip(*a_disp); anti_spiral_scatter.set_offsets(np.c_[[c+0.5 for c in asc],[r+0.5 for r in asr]])
            else: anti_spiral_scatter.set_offsets(np.empty((0,2)))
        spiral_scatter.set_visible(True); anti_spiral_scatter.set_visible(True); legend_singularities.set_visible(True)
    else:
        spiral_scatter.set_visible(False); anti_spiral_scatter.set_visible(False); legend_singularities.set_visible(False)

    # --- Update Track Lines Display ---
    for line in track_lines_plots_display: line.remove()
    track_lines_plots_display.clear()
    
    if current_show_tracks_view:
        for track_obj in current_tracks_for_display_plot:
            if track_obj['start_time'] <= current_display_time_idx_view <= track_obj['end_time']:
                points_to_draw_rc_z = []
                for i, t_point in enumerate(track_obj['time_indices']):
                    if t_point <= current_display_time_idx_view:
                        points_to_draw_rc_z.append(track_obj['points_rc_zoomed'][i])
                    else: break 
                
                if len(points_to_draw_rc_z) > 1:
                    color = config_module.SINGULARITY_EDGE_COLOR if track_obj['type'] == 'spiral' else config_module.SINGULARITY_ANTI_EDGE_COLOR
                    track_cols = [(p[1] + 0.5) for p in points_to_draw_rc_z]
                    track_rows = [(p[0] + 0.5) for p in points_to_draw_rc_z]
                    line, = ax.plot(track_cols, track_rows, color=color, linewidth=config_module.TRACK_LINEWIDTH, alpha=0.7, zorder=5)
                    track_lines_plots_display.append(line)
        
    fig.canvas.draw_idle()