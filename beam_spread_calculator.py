import numpy as np
import matplotlib.pyplot as plt

# --- Physical Constants ---
G_ACC = 9.80665  # Gravitational acceleration (m/s^2)
MOLAR_MASS_AIR = 0.0289644  # Molar mass of dry air (kg/mol)
R_GAS_CONST = 8.31447  # Universal gas constant (J/(mol*K))
P_STD_PA = 101325.0  # Standard pressure (Pa)
T_STD_EDLEN_K = 288.15  # Standard temperature for Edlen's formula (15°C in K)
DEFAULT_LAPSE_RATE = 0.0065 # K/m (positive for temperature DECREASE with height)

# --- Simulation Parameters ---
dist_wall_default = 500.0
n_beams_default = 10
d_angle_mrad_default = 0.1
offset_angle_mrad_default = 10.0  # Offset for beam angles in mrad
w_laser_nm_default = 660.0
d_step_ds_default = 0.1
h_limit_max_default = 30.0  # Physical limit for ray propagation
h_limit_min_default = 0.0   # Ground level
h_start_default = 2.0
h_plot_max_default = 3.0   # NEW: Default max height for plotting temperature profile

temp_at_ground_C_default = 20.0
gradient_type_default = "linear_lapse" # "constant", "linear_lapse", or "non_linear_ground_effect"

# Parameters for non-linear temperature profile
non_linear_delta_T_C_default = -5.0 # Temp change from ground to stable height (e.g. -5 means 5C cooler)
non_linear_decay_k_default = 1.0    # Decay constant k (1/m) for exponential change

# --- Helper Functions ---
def temp_air_profile(h, temp_at_ground_C=temp_at_ground_C_default,
                     gradient_type=gradient_type_default,
                     lapse_rate_K_per_m=DEFAULT_LAPSE_RATE,
                     non_linear_delta_T_C=non_linear_delta_T_C_default,
                     non_linear_decay_k=non_linear_decay_k_default):
    T0_K = temp_at_ground_C + 273.15
    if h < 0: h = 0

    if gradient_type == "constant":
        return T0_K
    elif gradient_type == "linear_lapse":
        T_K_at_h = T0_K - lapse_rate_K_per_m * h
        return T_K_at_h if T_K_at_h > 0 else 1.0
    elif gradient_type == "non_linear_ground_effect":
        T_K_at_h = T0_K + non_linear_delta_T_C * (1.0 - np.exp(-non_linear_decay_k * h))
        return T_K_at_h if T_K_at_h > 0 else 1.0
    else: # Default to linear lapse if type unknown
        T_K_at_h = T0_K - lapse_rate_K_per_m * h
        return T_K_at_h if T_K_at_h > 0 else 1.0

def pressure_at_height(h, temp_at_ground_K, p0_Pa=P_STD_PA, L_lapse_for_pressure=0.0):
    if h < 0: h = 0
    if temp_at_ground_K <= 0:
        return p0_Pa * np.exp(-G_ACC * MOLAR_MASS_AIR * h / (R_GAS_CONST * max(temp_at_ground_K, 1.0)))
    if L_lapse_for_pressure == 0:
        return p0_Pa * np.exp(-G_ACC * MOLAR_MASS_AIR * h / (R_GAS_CONST * temp_at_ground_K))
    else:
        base = 1.0 - (L_lapse_for_pressure * h) / temp_at_ground_K
        if base <= 0: return 1e-3
        exponent = (G_ACC * MOLAR_MASS_AIR) / (R_GAS_CONST * L_lapse_for_pressure)
        P_h = p0_Pa * (base ** exponent)
        return P_h if P_h > 0 else 1e-3

def refractive_index_air(T_K, P_Pa, lambda_vac_m):
    if T_K <= 0: T_K = 1.0;
    if P_Pa <=0: P_Pa = 1e-3
    lambda_vac_um = lambda_vac_m * 1e6; sigma_sq = (1 / lambda_vac_um)**2
    A = 8342.54; B = 2406147; C_val = 130; D = 15998; E_val = 38.9
    if abs(C_val - sigma_sq) < 1e-9 or abs(E_val - sigma_sq) < 1e-9: n_s_minus_1 = 2.73e-4
    else: n_s_minus_1_e8 = A + B / (C_val - sigma_sq) + D / (E_val - sigma_sq); n_s_minus_1 = n_s_minus_1_e8 / 1e8
    n_minus_1 = n_s_minus_1 * (P_Pa / P_STD_PA) * (T_STD_EDLEN_K / T_K)
    return 1.0 + n_minus_1

def calculate_dn_dh(h, temp_at_ground_C, p0_Pa, lambda_vac_m,
                    gradient_type, lapse_rate_K_per_m,
                    non_linear_delta_T_C, non_linear_decay_k,
                    delta_h_numerical=0.01):
    T_ground_K = temp_at_ground_C + 273.15
    temp_args_profile = {
        "temp_at_ground_C": temp_at_ground_C,
        "gradient_type": gradient_type,
        "lapse_rate_K_per_m": lapse_rate_K_per_m,
        "non_linear_delta_T_C": non_linear_delta_T_C,
        "non_linear_decay_k": non_linear_decay_k
    }
    L_for_pressure_in_n_calc = 0.0
    if gradient_type == "linear_lapse": L_for_pressure_in_n_calc = lapse_rate_K_per_m
    elif gradient_type == "non_linear_ground_effect": L_for_pressure_in_n_calc = 0.0

    def get_n_at_h(current_h):
        T_K = temp_air_profile(current_h, **temp_args_profile)
        P_Pa = pressure_at_height(current_h, T_ground_K, p0_Pa, L_for_pressure_in_n_calc)
        return refractive_index_air(T_K, P_Pa, lambda_vac_m)
    if h < delta_h_numerical and h >= 0:
        n_h = get_n_at_h(h); n_h_plus = get_n_at_h(h + delta_h_numerical)
        return (n_h_plus - n_h) / delta_h_numerical
    elif h < 0: return 0
    else:
        n_h_plus = get_n_at_h(h + delta_h_numerical); n_h_minus = get_n_at_h(h - delta_h_numerical)
        return (n_h_plus - n_h_minus) / (2 * delta_h_numerical)

# Add a new function to create a combined plot with 4 subplots
def create_combined_plot(
    all_ray_paths, final_y_positions, initial_angles_mrad,
    dist_wall_sim, d_step_ds_sim, h_limit_max_sim, h_limit_min_sim, h_start_sim,
    temp_at_ground_C_sim, gradient_type_sim, custom_lapse_rate_K_per_m,
    non_linear_delta_T_C_sim, non_linear_decay_k_sim, h_plot_max_sim
):
    """
    Creates a single figure with 4 subplots:
    1. Beam propagation paths
    2. Temperature profile 
    3. Temperature profile (with max height = h_plot_max_sim)
    4. Beam landing height differences (with vs. without temp gradient)
    """
    fig = plt.figure(figsize=(16, 12))  # Adjust height back to original
    
    # Create a 2x2 grid of subplots
    ax1 = plt.subplot(2, 2, 1)  # Top left: Ray Paths
    ax3 = plt.subplot(2, 2, 2)  # Top right: Temperature Profile
    ax2 = plt.subplot(2, 2, 3)  # Bottom left: Temperature Profile with limited height
    ax4 = plt.subplot(2, 2, 4)  # Bottom right: Landing height differences (previously ax5)
    
    # Figure 1: Ray Paths
    # Store straight-line landing heights for use in subplot 4
    no_gradient_landing_heights = []
    
    for i, (px, py) in enumerate(all_ray_paths):
        line = ax1.plot(px, py, label=f'Beam {i+1}')
        # Add straight-line reference trajectory
        if px and py:  # Make sure the beam has valid path data
            # Get the color of the current beam line
            beam_color = line[0].get_color()
            # Get the linewidth of the current beam line
            beam_linewidth = plt.rcParams['lines.linewidth'] if line[0].get_linewidth() == 1.0 else line[0].get_linewidth()
            # Calculate straight line trajectory
            angle_rad = initial_angles_mrad[i] * 1e-3  # Convert from mrad to rad
            y_at_wall = h_start_sim + np.tan(angle_rad) * dist_wall_sim
            no_gradient_landing_heights.append(y_at_wall)
            # Plot the straight reference line
            ax1.plot([0, dist_wall_sim], [h_start_sim, y_at_wall], 
                    color=beam_color, linestyle='--', 
                    linewidth=beam_linewidth*0.5,  # 50% thinner
                    alpha=0.5,  # 50% opacity
                    zorder=1)  # Place behind the actual beam paths
    
    ax1.axvline(x=dist_wall_sim, c='k', ls='--', label=f'Wall {dist_wall_sim}m')
    ax1.axhline(y=h_limit_max_sim, c='r', ls=':', label=f'Max H {h_limit_max_sim}m')
    ax1.axhline(y=h_limit_min_sim, c='g', ls=':', label=f'Ground {h_limit_min_sim}m')
    ax1.axhline(y=h_start_sim, c='purple', ls='-.', alpha=0.7, label=f'Start H {h_start_sim}m')
    
    # Mark impacts at wall
    for i, y_fp in enumerate(final_y_positions):
        lx = all_ray_paths[i][0][-1] if all_ray_paths[i][0] else 0
        if lx >= dist_wall_sim - d_step_ds_sim:
            if not((abs(y_fp-h_limit_max_sim)<1e-3 and lx<dist_wall_sim-d_step_ds_sim) or \
                (abs(y_fp-h_limit_min_sim)<1e-3 and lx<dist_wall_sim-d_step_ds_sim)):
                ax1.plot(dist_wall_sim, y_fp, 'ko', ms=5)
                
    
    ax1.set_xlabel('Distance (m)')
    ax1.set_ylabel('Height (m)')
    
    # Add minor grid lines on y axis with 0.1m spacing
    ax1.yaxis.set_minor_locator(plt.MultipleLocator(0.1))
    ax1.grid(True, which='major', alpha=0.7)
    ax1.grid(True, which='minor', axis='y', alpha=0.3, color='lightgray', linestyle=':')
    
    # Create more compact title for subplot 1
    fig1_title = ['1: Laser Beam Propagation']
    second_line = f'Ground T: {temp_at_ground_C_sim}°C, Start H: {h_start_sim}m'
    
    if gradient_type_sim == "linear_lapse":
        second_line += f', Lapse: {custom_lapse_rate_K_per_m:.4f} K/m'
    elif gradient_type_sim == "non_linear_ground_effect":
        second_line += f', ΔT={non_linear_delta_T_C_sim}°C, k={non_linear_decay_k_sim}/m'
    
    fig1_title.append(second_line)
    ax1.set_title('\n'.join(fig1_title), fontsize=16)
    ax1.legend(fontsize='small', loc='best')
    ax1.grid(True, alpha=0.7)
    
    # Set axis limits for subplot 1 and store plot_max_y for later use
    all_y_v = [y for _,p_y in all_ray_paths for y in p_y if p_y]
    plot_max_y = 0
    if all_y_v:
        plot_max_y = max(final_y_positions) + 0.5 if final_y_positions else max(all_y_v) + 1
        ax1.set_ylim(0, plot_max_y)
    else:
        # If there's no beam data, use h_limit_max_sim as fallback
        plot_max_y = h_limit_max_sim
    ax1.set_xlim(0, dist_wall_sim + dist_wall_sim * 0.05)
    
    # Prepare temperature profile data for both subplot 2 and 3
    heights = np.linspace(0, h_plot_max_sim, 100)
    temp_args = {
        "temp_at_ground_C": temp_at_ground_C_sim,
        "gradient_type": gradient_type_sim,
        "lapse_rate_K_per_m": custom_lapse_rate_K_per_m,
        "non_linear_delta_T_C": non_linear_delta_T_C_sim,
        "non_linear_decay_k": non_linear_decay_k_sim
    }
    temperatures_K = [temp_air_profile(h, **temp_args) for h in heights]
    temperatures_C = [T - 273.15 for T in temperatures_K]
    
    # Common temperature profile plotting function with compact title
    def plot_temp_profile(ax, title_number, max_height):
        ax.plot(temperatures_C, heights, 'b-', label='Temperature (°C)')
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax.set_ylabel('Height (m)')
        ax.set_xlabel('Temperature (°C)')
        
        # Create more compact title for temperature profile plots
        title_parts = [f'{title_number}: Temperature Profile']
        second_line = f'Ground Temp: {temp_at_ground_C_sim}°C'
        
        if gradient_type_sim == "linear_lapse":
            second_line += f', Lapse Rate: {custom_lapse_rate_K_per_m:.4f} K/m'
        elif gradient_type_sim == "non_linear_ground_effect":
            second_line += f', ΔT={non_linear_delta_T_C_sim}°C, k={non_linear_decay_k_sim}/m'
        
        title_parts.append(second_line)
        ax.set_title('\n'.join(title_parts), fontsize=16)
        
        temp_at_ground_C = temp_air_profile(0, **temp_args) - 273.15
        temp_at_h_start_C = temp_air_profile(h_start_sim, **temp_args) - 273.15
        ax.plot(temp_at_ground_C, 0, 'go', markersize=6, label=f'Ground Level: {temp_at_ground_C:.2f}°C')
        if 0 <= h_start_sim <= max_height:
            ax.plot(temp_at_h_start_C, h_start_sim, 'ro', markersize=6, 
                    label=f'Instrument Height: {temp_at_h_start_C:.2f}°C')
            ax.axhline(y=h_start_sim, color='r', linestyle=':', alpha=0.5)
        
        ax.legend(loc='best', fontsize='small')
        ax.grid(True)
        ax.set_ylim(0, max_height)
    
    # Figure 2: Temperature Profile (Max height matches subplot 1)
    plot_temp_profile(ax3, 2, plot_max_y)
    
    # Figure 3: Temperature Profile (Max height = h_plot_max_sim)  
    plot_temp_profile(ax2, 3, h_plot_max_sim)
    
    # Figure 4: Beam Landing Height Differences (previously Figure 5)
    if final_y_positions and no_gradient_landing_heights and len(final_y_positions) == len(no_gradient_landing_heights):
        # Calculate height differences (with gradient - without gradient) in mm
        height_differences_mm = [(y_with - y_without) * 1000 for y_with, y_without in zip(final_y_positions, no_gradient_landing_heights)]
        
        # Plot differences vs. straight-line landing heights
        ax4.plot(no_gradient_landing_heights, height_differences_mm, 'o-', color='darkblue', linewidth=1.5, markersize=6)
        
        # Add horizontal line at y=0 (no difference)
        ax4.axhline(y=0, color='gray', linestyle='-', alpha=0.5, linewidth=1)
        
        # Label each point with beam number
        for i, (x, y) in enumerate(zip(no_gradient_landing_heights, height_differences_mm)):
            ax4.annotate(f"B{i+1}", (x, y), fontsize=9, 
                        xytext=(5, 5), textcoords='offset points',
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.7))
        
        ax4.set_xlabel('Theoretical Landing Height without Temperature Gradient (m)')
        ax4.set_ylabel('Height Difference: With Gradient - Without Gradient (mm)')
        ax4.set_title('4: Beam Bending Effect due to Temperature Gradient', fontsize=16)
        
        ax4.grid(True, which='major', alpha=0.7)
        ax4.grid(True, which='minor', alpha=0.3, linestyle=':')
        ax4.minorticks_on()
    
    plt.tight_layout()
    # Keep title at the same high position but create more room below it
    fig.suptitle(f"Atmospheric Effect on Laser Beam ({gradient_type_sim})", fontsize=16, y=0.998, weight='bold')
    # Decrease the top value to move subplots down and create more space for the title
    plt.subplots_adjust(top=0.92)
    
    return fig

# Modify the run_simulation function to use the combined plot
def run_simulation(
    dist_wall_sim=dist_wall_default,
    n_beams_sim=n_beams_default,
    d_angle_mrad_sim=d_angle_mrad_default,
    offset_angle_mrad_sim=offset_angle_mrad_default,  # Add the offset parameter
    custom_initial_angles_rad_sim=None,
    w_laser_nm_sim=w_laser_nm_default,
    d_step_ds_sim=d_step_ds_default,
    h_limit_max_sim=h_limit_max_default,    # Physical boundary for rays
    h_limit_min_sim=h_limit_min_default,
    h_start_sim=h_start_default,
    h_plot_max_sim=h_plot_max_default,      # Max height for temp profile plot
    temp_at_ground_C_sim=temp_at_ground_C_default,
    gradient_type_sim=gradient_type_default,
    custom_lapse_rate_K_per_m=DEFAULT_LAPSE_RATE, # For linear_lapse
    non_linear_delta_T_C_sim=non_linear_delta_T_C_default, # For non_linear_ground_effect
    non_linear_decay_k_sim=non_linear_decay_k_default,   # For non_linear_ground_effect
    p0_Pa_sim=P_STD_PA
    ):

    print(f"\n--- Starting Simulation ---")
    print(f"Laser Start Height: {h_start_sim}m, Plot Max Height for Temp Profile: {h_plot_max_sim}m")
    print(f"Wall Dist: {dist_wall_sim}m | Num Beams: {n_beams_sim} | Ang. Sep: {d_angle_mrad_sim}mrad | Offset: {offset_angle_mrad_sim}mrad | Lambda: {w_laser_nm_sim}nm")
    print(f"Ground Temp: {temp_at_ground_C_sim}°C | Temp Grad: '{gradient_type_sim}'")

    L_for_pressure_sim = 0.0 
    if gradient_type_sim == "linear_lapse":
        L_for_pressure_sim = custom_lapse_rate_K_per_m
        print(f"  Linear Lapse Rate: {custom_lapse_rate_K_per_m} K/m")
        print(f"  Pressure calc using lapse rate: {L_for_pressure_sim} K/m")
    elif gradient_type_sim == "non_linear_ground_effect":
        L_for_pressure_sim = 0.0
        print(f"  Non-linear: Delta_T={non_linear_delta_T_C_sim}°C, k={non_linear_decay_k_sim}/m")
        print(f"  Pressure calc using ISOTHERMAL model (L_lapse=0 K/m based on T_ground).")
    else:
        print(f"  Pressure calc using ISOTHERMAL model (L_lapse=0 K/m based on T_ground).")

    lambda_vac_m_sim = w_laser_nm_sim * 1e-9; d_angle_rad_sim = d_angle_mrad_sim * 1e-3
    offset_angle_rad_sim = offset_angle_mrad_sim * 1e-3  # Convert offset angle to radians
    
    if custom_initial_angles_rad_sim is not None:
        initial_angles_rad = np.asarray(custom_initial_angles_rad_sim)
        if n_beams_sim != len(initial_angles_rad) and len(initial_angles_rad) > 0 :
            n_beams_sim = len(initial_angles_rad)
        elif len(initial_angles_rad) == 0:
            initial_angles_rad = np.linspace(-(n_beams_sim -1)*d_angle_rad_sim/2., (n_beams_sim-1)*d_angle_rad_sim/2., max(1,n_beams_sim))
            initial_angles_rad += offset_angle_rad_sim  # Add offset angle when generating default angles
    else:
        if n_beams_sim <=0: n_beams_sim=1; initial_angles_rad=np.array([offset_angle_rad_sim])  # Apply offset to single beam
        else: 
            initial_angles_rad = np.linspace(-(n_beams_sim-1)*d_angle_rad_sim/2.,(n_beams_sim-1)*d_angle_rad_sim/2., n_beams_sim)
            initial_angles_rad += offset_angle_rad_sim  # Add offset angle to generated angles
    
    initial_angles_mrad = initial_angles_rad * 1000
    all_ray_paths = []; final_y_positions = []

    temp_profile_args_for_sim = {
        "temp_at_ground_C": temp_at_ground_C_sim,
        "gradient_type": gradient_type_sim,
        "lapse_rate_K_per_m": custom_lapse_rate_K_per_m,
        "non_linear_delta_T_C": non_linear_delta_T_C_sim,
        "non_linear_decay_k": non_linear_decay_k_sim
    }

    for i_beam, phi_initial in enumerate(initial_angles_rad):
        x, y = 0.0, h_start_sim; phi = phi_initial
        if y > h_limit_max_sim or y < h_limit_min_sim:
            print(f"Beam {i_beam+1} starts outside vertical limits. Terminating.")
            limit_val = h_limit_max_sim if y > h_limit_max_sim else h_limit_min_sim
            all_ray_paths.append(([x],[limit_val])); final_y_positions.append(limit_val)
            continue
        current_path_x = [x]; current_path_y = [y]
        total_s_propagated = 0.0; terminated_early = False
        while x < dist_wall_sim:
            T_curr_K = temp_air_profile(y, **temp_profile_args_for_sim)
            T_ground_K_for_pressure = temp_at_ground_C_sim + 273.15
            P_curr_Pa = pressure_at_height(y, T_ground_K_for_pressure, p0_Pa_sim, L_for_pressure_sim)
            if T_curr_K <= 0 or P_curr_Pa <= 0: terminated_early = True; break
            n_curr = refractive_index_air(T_curr_K, P_curr_Pa, lambda_vac_m_sim)
            grad_n_y = calculate_dn_dh(y, temp_at_ground_C_sim, p0_Pa_sim, lambda_vac_m_sim,
                                       gradient_type_sim, custom_lapse_rate_K_per_m,
                                       non_linear_delta_T_C_sim, non_linear_decay_k_sim)
            cos_phi = np.cos(phi); sin_phi = np.sin(phi)
            dphi_ds = (1.0 / n_curr) * grad_n_y * cos_phi
            x += cos_phi * d_step_ds_sim; y += sin_phi * d_step_ds_sim
            phi += dphi_ds * d_step_ds_sim; total_s_propagated += d_step_ds_sim
            current_path_x.append(x); current_path_y.append(y)
            if y > h_limit_max_sim: current_path_y[-1] = h_limit_max_sim; terminated_early = True; break
            if y < h_limit_min_sim: current_path_y[-1] = h_limit_min_sim; terminated_early = True; break
            if total_s_propagated > 2 * (dist_wall_sim + h_limit_max_sim + abs(h_limit_min_sim)): terminated_early = True; break
        all_ray_paths.append((current_path_x, current_path_y))
        final_y_at_termination = current_path_y[-1]
        if not terminated_early and x >= dist_wall_sim and len(current_path_x) > 1:
            try:
                final_y_at_termination = np.interp(dist_wall_sim, current_path_x, current_path_y)
                final_y_at_termination = max(h_limit_min_sim, min(h_limit_max_sim, final_y_at_termination))
            except Exception: pass
        final_y_positions.append(final_y_at_termination)
    create_combined_plot(
        all_ray_paths, final_y_positions, initial_angles_mrad,
        dist_wall_sim, d_step_ds_sim, h_limit_max_sim, h_limit_min_sim, h_start_sim,
        temp_at_ground_C_sim, gradient_type_sim, custom_lapse_rate_K_per_m,
        non_linear_delta_T_C_sim, non_linear_decay_k_sim, h_plot_max_sim
    )
    plt.show()
    print("\n--- Final Y positions (at termination: wall, ground, or max height) ---")
    for i,y_pos in enumerate(final_y_positions):
        status = "";lx = all_ray_paths[i][0][-1] if all_ray_paths[i] and all_ray_paths[i][0] else -1
        if abs(y_pos-h_limit_max_sim)<1e-3:status=f"(Max H {h_limit_max_sim}m)"
        elif abs(y_pos-h_limit_min_sim)<1e-3:status=f"(Ground {h_limit_min_sim}m)"
        elif lx>=dist_wall_sim-d_step_ds_sim:status=f"(Wall x~{dist_wall_sim}m)"
        else:status=f"(Early x={lx:.2f}m)"
        print(f"B{i+1}(Ang:{initial_angles_mrad[i]:.3f}mrad):Y={y_pos:.4f}m {status}")
    return all_ray_paths, final_y_positions, initial_angles_mrad

if __name__ == '__main__':
    print("\n### Running Simulation with Non-Linear Ground Effect Temperature Profile ###")
    run_simulation(
        h_start_sim=1.5,
        temp_at_ground_C_sim=25.0,
        gradient_type_sim="non_linear_ground_effect",
        non_linear_delta_T_C_sim = -5.0,
        non_linear_decay_k_sim = 0.8,
        n_beams_sim=7,
        d_angle_mrad_sim=0.5,
        offset_angle_mrad_sim=-1.0,  # Specify the offset angle
        dist_wall_sim=500,
        h_limit_min_sim=0.0,
        h_limit_max_sim=40.0,
        h_plot_max_sim=10.0
    )