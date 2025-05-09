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
        # T(h) = T_ground - DeltaT * (1 - exp(-k*h))
        # DeltaT is the total change. If non_linear_delta_T_C is negative (cooler),
        # then T0_K - (negative_delta) * (positive_term) = T0_K + positive_change_term. This is wrong.
        # It should be T(h) = T_ground + non_linear_delta_T_C * (1 - exp(-k*h)) if delta_T is difference
        # Or T(h) = T_stable + (T_ground - T_stable) * exp(-k*h)
        # Let T_stable = T0_K + non_linear_delta_T_C (where non_linear_delta_T_C is the diff T_stable - T0_K)
        # T_K_at_h = (T0_K + non_linear_delta_T_C) + (T0_K - (T0_K + non_linear_delta_T_C)) * np.exp(-non_linear_decay_k * h)
        # T_K_at_h = T0_K + non_linear_delta_T_C - non_linear_delta_T_C * np.exp(-non_linear_decay_k * h)
        # T_K_at_h = T0_K + non_linear_delta_T_C * (1.0 - np.exp(-non_linear_decay_k * h))
        # non_linear_delta_T_C is the difference (T_stable - T_ground).
        # So if ground=20C (293.15K), stable=-5C cooler (15C, 288.15K), delta_T_C = -5.
        # T(h) = 293.15K + (-5K) * (1 - exp(-kh))
        T_K_at_h = T0_K + non_linear_delta_T_C * (1.0 - np.exp(-non_linear_decay_k * h))
        return T_K_at_h if T_K_at_h > 0 else 1.0
    else: # Default to linear lapse if type unknown
        T_K_at_h = T0_K - lapse_rate_K_per_m * h
        return T_K_at_h if T_K_at_h > 0 else 1.0

def pressure_at_height(h, temp_at_ground_K, p0_Pa=P_STD_PA, L_lapse_for_pressure=0.0):
    # (Implementation from previous response)
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
    # (Implementation from previous response)
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
                    non_linear_delta_T_C, non_linear_decay_k, # Added non-linear params
                    delta_h_numerical=0.01):
    T_ground_K = temp_at_ground_C + 273.15
    temp_args_profile = {
        "temp_at_ground_C": temp_at_ground_C,
        "gradient_type": gradient_type,
        "lapse_rate_K_per_m": lapse_rate_K_per_m,
        "non_linear_delta_T_C": non_linear_delta_T_C,       # Pass through
        "non_linear_decay_k": non_linear_decay_k         # Pass through
    }
    L_for_pressure_in_n_calc = 0.0
    if gradient_type == "linear_lapse": L_for_pressure_in_n_calc = lapse_rate_K_per_m
    elif gradient_type == "non_linear_ground_effect": L_for_pressure_in_n_calc = 0.0 # Isothermal pressure for non-linear T

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

# --- MODIFIED PLOTTING FUNCTIONS ---
def plot_figure3_temperature_vs_height(h_start_laser, temp_at_ground_C, gradient_type,
                                       lapse_rate_K_per_m, # For linear type
                                       non_linear_delta_T_C, non_linear_decay_k, # For non-linear
                                       h_plot_max_actual, figure_num="3"): # Use h_plot_max_actual
    """
    Figure 3: Plots air temperature (Celsius and Kelvin) as a function of height up to h_plot_max_actual.
    """
    heights_plot = np.linspace(0, h_plot_max_actual, 200) # Use h_plot_max_actual
    temp_args = {
        "temp_at_ground_C": temp_at_ground_C, "gradient_type": gradient_type,
        "lapse_rate_K_per_m": lapse_rate_K_per_m,
        "non_linear_delta_T_C": non_linear_delta_T_C,
        "non_linear_decay_k": non_linear_decay_k
    }
    temperatures_K_profile = [temp_air_profile(h, **temp_args) for h in heights_plot]
    temperatures_C_profile = [T - 273.15 for T in temperatures_K_profile]

    temp_at_sea_level_C = temp_air_profile(0, **temp_args) - 273.15
    temp_at_h_start_C = temp_air_profile(h_start_laser, **temp_args) - 273.15

    plt.figure(figsize=(7, 9))
    ax1 = plt.gca()
    ax1.plot(temperatures_C_profile, heights_plot, 'b-', label=f'Temp. Profile ({gradient_type})')
    ax1.set_xlabel('Temperature (°C)'); ax1.set_ylabel('Height above Ground (m)')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.plot(temp_at_sea_level_C, 0, 'go', markersize=8, label=f'Sea Level (0m): {temp_at_sea_level_C:.2f}°C')
    if 0 <= h_start_laser <= h_plot_max_actual: # Check against plot max for visibility
        ax1.plot(temp_at_h_start_C, h_start_laser, 'ro', markersize=8, label=f'Laser Start ({h_start_laser}m): {temp_at_h_start_C:.2f}°C')
        ax1.axhline(y=h_start_laser, color='r', linestyle=':', alpha=0.5, label=f'_nolegend_')

    ax2 = ax1.twiny()
    ax2.plot([T_K + 273.15 for T_K in temperatures_C_profile], heights_plot, color='None') # Sync limits
    ax2.set_xlabel('Temperature (K)')
    handles1, labels1 = ax1.get_legend_handles_labels()
    ax1.legend(handles1, labels1, loc='best')

    title_parts = [f'Figure {figure_num}: Air Temperature vs. Height', f'Ground (0m) Temp: {temp_at_ground_C}°C']
    if gradient_type == "linear_lapse":
        title_parts.append(f'Lapse Rate: {lapse_rate_K_per_m:.4f} K/m')
    elif gradient_type == "non_linear_ground_effect":
        title_parts.append(f'Non-linear: Delta_T={non_linear_delta_T_C}°C, k={non_linear_decay_k}/m')
    plt.title('\n'.join(title_parts), pad=20)
    plt.tight_layout(rect=[0, 0, 1, 0.96])


def plot_figure4_beam_landing_separation(final_y_positions, initial_angles_mrad, figure_num="4"):
    if not final_y_positions or len(final_y_positions) < 2:
        print(f"Figure {figure_num}: Not enough beam landing positions to calculate separations.")
        return
    separations = []; pair_labels = []
    for i in range(len(final_y_positions) - 1):
        sep = final_y_positions[i+1] - final_y_positions[i]; separations.append(sep)
        label = f"B{i+1}-B{i+2}"
        if initial_angles_mrad is not None and len(initial_angles_mrad) == len(final_y_positions):
             label += f"\n({initial_angles_mrad[i]:.2f} to {initial_angles_mrad[i+1]:.2f} mrad)"
        pair_labels.append(label)
    if not separations: print(f"Figure {figure_num}: No separations calculated."); return
    avg_separation = np.mean(separations)
    plt.figure(figsize=(max(10, len(separations)*0.8), 7))
    x_indices = np.arange(len(separations))
    plt.bar(x_indices, separations, color='coral', label='Separation (m)')
    plt.axhline(avg_separation, color='dodgerblue', linestyle='--', linewidth=2, label=f'Avg Sep: {avg_separation:.4f} m')
    plt.xlabel('Adjacent Beam Pair'); plt.ylabel('Landing Height Separation (m)')
    plt.title(f'Figure {figure_num}: Difference in Landing Height Between Adjacent Beams')
    if pair_labels: plt.xticks(x_indices, pair_labels, rotation=45, ha="right", fontsize=9)
    plt.legend(); plt.grid(True, axis='y', alpha=0.7); plt.tight_layout()
    # Adjust y-axis limits to center around the average value and show the full range of bars
    if separations:
        max_sep = max(separations)
        min_sep = min(separations)
        plt.ylim(min_sep*0.99, max_sep*1.01)
        # Add tick marks for each 0.001
        plt.yticks(np.arange(min_sep, max_sep + 0.001, 0.001))

    # Add labels to each bar showing the landing height separation
    for i, sep in enumerate(separations):
        plt.text(x_indices[i], sep, f"{sep:.4f}", ha='center', va='bottom', fontsize=9)


# --- Main Simulation ---
def run_simulation(
    dist_wall_sim=dist_wall_default,
    n_beams_sim=n_beams_default,
    d_angle_mrad_sim=d_angle_mrad_default,
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

    # ... (Initial checks for h_start_sim, print statements) ...
    print(f"\n--- Starting Simulation ---")
    print(f"Laser Start Height: {h_start_sim}m, Plot Max Height for Temp Profile: {h_plot_max_sim}m")
    print(f"Wall Dist: {dist_wall_sim}m | Num Beams: {n_beams_sim} | Ang. Sep: {d_angle_mrad_sim}mrad | Lambda: {w_laser_nm_sim}nm")
    print(f"Ground Temp: {temp_at_ground_C_sim}°C | Temp Grad: '{gradient_type_sim}'")

    L_for_pressure_sim = 0.0
    if gradient_type_sim == "linear_lapse":
        L_for_pressure_sim = custom_lapse_rate_K_per_m
        print(f"  Linear Lapse Rate: {custom_lapse_rate_K_per_m} K/m")
        print(f"  Pressure calc using lapse rate: {L_for_pressure_sim} K/m")
    elif gradient_type_sim == "non_linear_ground_effect":
        L_for_pressure_sim = 0.0 # Isothermal pressure calculation as simplification
        print(f"  Non-linear: Delta_T={non_linear_delta_T_C_sim}°C, k={non_linear_decay_k_sim}/m")
        print(f"  Pressure calc using ISOTHERMAL model (L_lapse=0 K/m based on T_ground).")
    else: # Constant temperature
        print(f"  Pressure calc using ISOTHERMAL model (L_lapse=0 K/m based on T_ground).")
    # ... (rest of setup: lambda, initial_angles_rad, etc. from previous version) ...
    lambda_vac_m_sim = w_laser_nm_sim * 1e-9; d_angle_rad_sim = d_angle_mrad_sim * 1e-3
    if custom_initial_angles_rad_sim is not None:
        initial_angles_rad = np.asarray(custom_initial_angles_rad_sim)
        if n_beams_sim != len(initial_angles_rad) and len(initial_angles_rad) > 0 :
            n_beams_sim = len(initial_angles_rad)
        elif len(initial_angles_rad) == 0:
            initial_angles_rad = np.linspace(-(n_beams_sim -1)*d_angle_rad_sim/2., (n_beams_sim-1)*d_angle_rad_sim/2., max(1,n_beams_sim))
    else:
        if n_beams_sim <=0: n_beams_sim=1; initial_angles_rad=np.array([0.0])
        else: initial_angles_rad = np.linspace(-(n_beams_sim-1)*d_angle_rad_sim/2.,(n_beams_sim-1)*d_angle_rad_sim/2., n_beams_sim)
    initial_angles_mrad = initial_angles_rad * 1000
    all_ray_paths = []; final_y_positions = []

    temp_profile_args_for_sim = { # Arguments for temp_air_profile within simulation loop & dn/dh
        "temp_at_ground_C": temp_at_ground_C_sim,
        "gradient_type": gradient_type_sim,
        "lapse_rate_K_per_m": custom_lapse_rate_K_per_m,
        "non_linear_delta_T_C": non_linear_delta_T_C_sim,
        "non_linear_decay_k": non_linear_decay_k_sim
    }

    for i_beam, phi_initial in enumerate(initial_angles_rad):
        x, y = 0.0, h_start_sim; phi = phi_initial
        # ... (beam start checks from previous version) ...
        if y > h_limit_max_sim or y < h_limit_min_sim: # Simplified start check
            print(f"Beam {i_beam+1} starts outside vertical limits. Terminating.")
            limit_val = h_limit_max_sim if y > h_limit_max_sim else h_limit_min_sim
            all_ray_paths.append(([x],[limit_val])); final_y_positions.append(limit_val)
            continue

        current_path_x = [x]; current_path_y = [y]
        total_s_propagated = 0.0; terminated_early = False

        while x < dist_wall_sim:
            T_curr_K = temp_air_profile(y, **temp_profile_args_for_sim) # Use dictionary
            T_ground_K_for_pressure = temp_at_ground_C_sim + 273.15
            P_curr_Pa = pressure_at_height(y, T_ground_K_for_pressure, p0_Pa_sim, L_for_pressure_sim)

            if T_curr_K <= 0 or P_curr_Pa <= 0: terminated_early = True; break
            n_curr = refractive_index_air(T_curr_K, P_curr_Pa, lambda_vac_m_sim)
            # Pass all necessary temp params to calculate_dn_dh
            grad_n_y = calculate_dn_dh(y, temp_at_ground_C_sim, p0_Pa_sim, lambda_vac_m_sim,
                                       gradient_type_sim, custom_lapse_rate_K_per_m,
                                       non_linear_delta_T_C_sim, non_linear_decay_k_sim) # Pass non-linear params
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
            try: # Interpolate at wall
                final_y_at_termination = np.interp(dist_wall_sim, current_path_x, current_path_y)
                final_y_at_termination = max(h_limit_min_sim, min(h_limit_max_sim, final_y_at_termination))
            except Exception: pass # Keep last y if interp fails
        final_y_positions.append(final_y_at_termination)

    # --- Plotting ---
    # Figure 1: Ray Paths (plotting logic largely same, ensure title reflects profile)
    plt.figure(figsize=(12, 8))
    for i, (px, py) in enumerate(all_ray_paths): plt.plot(px, py, label=f'B{i+1} ({initial_angles_mrad[i]:.2f} mrad)')
    plt.axvline(x=dist_wall_sim, c='k', ls='--', label=f'Wall {dist_wall_sim}m')
    plt.axhline(y=h_limit_max_sim, c='r', ls=':', label=f'Max H {h_limit_max_sim}m')
    plt.axhline(y=h_limit_min_sim, c='g', ls=':', label=f'Ground {h_limit_min_sim}m')
    plt.axhline(y=h_start_sim, c='purple', ls='-.', alpha=0.7, label=f'Start H {h_start_sim}m')
    # (Mark impacts logic from previous)
    for i, y_fp in enumerate(final_y_positions):
        lx = all_ray_paths[i][0][-1] if all_ray_paths[i][0] else 0
        if lx >= dist_wall_sim - d_step_ds_sim:
            if not((abs(y_fp-h_limit_max_sim)<1e-3 and lx<dist_wall_sim-d_step_ds_sim) or \
                   (abs(y_fp-h_limit_min_sim)<1e-3 and lx<dist_wall_sim-d_step_ds_sim)):
                plt.plot(dist_wall_sim, y_fp, 'ko', ms=5)
    plt.xlabel('Distance (m)'); plt.ylabel('Height (m)')
    fig1_title = [f'Figure 1: Laser Beam Propagation ({gradient_type_sim})',
                  f'Ground T: {temp_at_ground_C_sim}°C, Start H: {h_start_sim}m']
    if gradient_type_sim == "linear_lapse": fig1_title.append(f'Lapse: {custom_lapse_rate_K_per_m:.4f} K/m')
    elif gradient_type_sim == "non_linear_ground_effect": fig1_title.append(f'Non-linear: ΔT={non_linear_delta_T_C_sim}°C, k={non_linear_decay_k_sim}/m')
    plt.title('\n'.join(fig1_title)); plt.legend(fontsize='small', loc='best'); plt.grid(True, alpha=0.7)
    # (ylim/xlim logic from previous)
    all_y_v = [y for _,p_y in all_ray_paths for y in p_y if p_y]
    if all_y_v:
        min_y_d, max_y_d = min(all_y_v), max(all_y_v)
        plot_min_y = min(h_limit_min_sim, min_y_d, h_start_sim)-1
        plot_max_y = max(h_limit_max_sim, max_y_d, h_start_sim)+1
        # Respect h_plot_max_sim for Figure 1's YLIMS if it's smaller than physical limits and data.
        # This is tricky for ray path plot - usually want to see all data. Let's prioritize data + h_limit_max_sim.
        # h_plot_max_sim is primarily for Figure 3.
        plt.ylim(max(h_limit_min_sim-2, plot_min_y), min(h_limit_max_sim+5, plot_max_y)) # Increased upper padding potential
    plt.xlim(0, dist_wall_sim + dist_wall_sim * 0.05)


    # Figure 2: Landing Height vs. Initial Angle (Plotting logic same)
    # (Code from previous response, ensure title/labels are clear)
    valid_fy = []; valid_ia = []
    for i, y_fp in enumerate(final_y_positions):
        lx = all_ray_paths[i][0][-1] if all_ray_paths[i][0] else 0
        is_early_max = (abs(y_fp-h_limit_max_sim)<1e-3 and lx<dist_wall_sim-d_step_ds_sim)
        is_early_min = (abs(y_fp-h_limit_min_sim)<1e-3 and lx<dist_wall_sim-d_step_ds_sim)
        if lx >= dist_wall_sim - d_step_ds_sim and not is_early_max and not is_early_min:
            valid_fy.append(y_fp); valid_ia.append(initial_angles_mrad[i])
    if valid_fy:
        plt.figure(figsize=(8,6)); plt.plot(valid_ia, valid_fy, 'bo-', label="Impact on Wall")
        plt.xlabel("Initial Angle (mrad)"); plt.ylabel(f"Final Height at Wall ({dist_wall_sim}m)")
        plt.title("Fig 2: Beam Landing Height at Wall vs. Initial Angle"); plt.grid(True,alpha=0.7); plt.legend()


    # Figure 3: Temperature Profile (Pass h_plot_max_sim)
    plot_figure3_temperature_vs_height(h_start_sim, temp_at_ground_C_sim, gradient_type_sim,
                                       custom_lapse_rate_K_per_m,    # For linear
                                       non_linear_delta_T_C_sim,     # For non-linear
                                       non_linear_decay_k_sim,       # For non-linear
                                       h_plot_max_sim,               # Use h_plot_max_sim
                                       figure_num="3")

    # Figure 4: Beam Landing Separations (Plotting logic same)
    plot_figure4_beam_landing_separation(final_y_positions, initial_angles_mrad, figure_num="4")

    plt.show()

    # (Final Y positions printout from previous response)
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
    # run_simulation( # Default
    #     h_start_sim=2.0, temp_at_ground_C_sim=20.0, gradient_type_sim="linear_lapse",
    #     custom_lapse_rate_K_per_m=0.0065, n_beams_sim=11, d_angle_mrad_sim=0.05,
    #     dist_wall_sim=1000, h_limit_min_sim=0.0, h_plot_max_sim=30.0 # Example h_plot_max
    # )
    # run_simulation( # Constant Temp
    #     h_start_sim=0.5, temp_at_ground_C_sim=15.0, gradient_type_sim="constant",
    #     custom_lapse_rate_K_per_m=0.0, n_beams_sim=7, d_angle_mrad_sim=0.1,
    #     dist_wall_sim=500, h_limit_min_sim=0.0, h_plot_max_sim=10.0
    # )
    # run_simulation( # Inversion
    #     h_start_sim=1.0, temp_at_ground_C_sim=10.0, gradient_type_sim="linear_lapse",
    #     custom_lapse_rate_K_per_m=-0.05, h_limit_max_sim=50, n_beams_sim=9,
    #     d_angle_mrad_sim=0.08, dist_wall_sim=1500, h_limit_min_sim=0.0, h_plot_max_sim=50.0
    # )
    print("\n### Running Simulation with Non-Linear Ground Effect Temperature Profile ###")
    run_simulation(
        h_start_sim=1.5,
        temp_at_ground_C_sim=25.0,
        gradient_type_sim="non_linear_ground_effect",
        non_linear_delta_T_C_sim = -10.0, # Gets 10C cooler from ground to stable height
        non_linear_decay_k_sim = 0.8,   # Decay constant (1/m)
        n_beams_sim=7,
        d_angle_mrad_sim=0.05,
        dist_wall_sim=800,
        h_limit_min_sim=0.0,
        h_limit_max_sim=40.0, # Physical limit
        h_plot_max_sim=15.0   # Plot temp profile up to 15m to see detail near ground
    )