import numpy as np
import matplotlib.pyplot as plt

# --- Physical Constants ---
G_ACC = 9.80665  # Gravitational acceleration (m/s^2)
MOLAR_MASS_AIR = 0.0289644  # Molar mass of dry air (kg/mol)
R_GAS_CONST = 8.31447  # Universal gas constant (J/(mol*K))
P_STD_PA = 101325.0  # Standard pressure (Pa)
T_STD_EDLEN_K = 288.15  # Standard temperature for Edlen's formula (15°C in K)
DEFAULT_LAPSE_RATE = 0.0065

# --- Simulation Parameters ---
dist_wall_default = 500.0
n_beams_default = 10
d_angle_mrad_default = 0.1
w_laser_nm_default = 660.0
d_step_ds_default = 0.1
h_limit_max_default = 30.0
h_limit_min_default = 0.0
h_start_default = 2.0
temp_at_ground_C_default = 20.0
gradient_type_default = "linear_lapse"

# --- Helper Functions --- (temp_air_profile, pressure_at_height, refractive_index_air, calculate_dn_dh - kept as in your previous correct version)
def temp_air_profile(h, temp_at_ground_C=temp_at_ground_C_default,
                     gradient_type=gradient_type_default,
                     lapse_rate_K_per_m=DEFAULT_LAPSE_RATE):
    T0_K = temp_at_ground_C + 273.15
    if h < 0: h = 0
    if gradient_type == "constant":
        return T0_K
    elif gradient_type == "linear_lapse":
        T_K_at_h = T0_K - lapse_rate_K_per_m * h
        return T_K_at_h if T_K_at_h > 0 else 1.0
    else:
        T_K_at_h = T0_K - lapse_rate_K_per_m * h
        return T_K_at_h if T_K_at_h > 0 else 1.0

def pressure_at_height(h, temp_at_ground_K, p0_Pa=P_STD_PA,
                       L_lapse_for_pressure=0.0):
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
    if T_K <= 0: T_K = 1.0
    if P_Pa <=0: P_Pa = 1e-3
    lambda_vac_um = lambda_vac_m * 1e6
    sigma_sq = (1 / lambda_vac_um)**2
    A = 8342.54; B = 2406147; C_val = 130; D = 15998; E_val = 38.9
    if abs(C_val - sigma_sq) < 1e-9 or abs(E_val - sigma_sq) < 1e-9:
        n_s_minus_1 = 2.73e-4
    else:
        n_s_minus_1_e8 = A + B / (C_val - sigma_sq) + D / (E_val - sigma_sq)
        n_s_minus_1 = n_s_minus_1_e8 / 1e8
    n_minus_1 = n_s_minus_1 * (P_Pa / P_STD_PA) * (T_STD_EDLEN_K / T_K)
    return 1.0 + n_minus_1

def calculate_dn_dh(h, temp_at_ground_C, p0_Pa, lambda_vac_m,
                    gradient_type, lapse_rate_K_per_m,
                    delta_h_numerical=0.01):
    T_ground_K = temp_at_ground_C + 273.15
    temp_args_profile = {
        "temp_at_ground_C": temp_at_ground_C,
        "gradient_type": gradient_type,
        "lapse_rate_K_per_m": lapse_rate_K_per_m
    }
    L_for_pressure_in_n_calc = 0.0
    if gradient_type == "linear_lapse":
        L_for_pressure_in_n_calc = lapse_rate_K_per_m
    def get_n_at_h(current_h):
        T_K = temp_air_profile(current_h, **temp_args_profile)
        P_Pa = pressure_at_height(current_h, T_ground_K, p0_Pa, L_for_pressure_in_n_calc)
        return refractive_index_air(T_K, P_Pa, lambda_vac_m)
    if h < delta_h_numerical and h >= 0:
        n_h = get_n_at_h(h)
        n_h_plus = get_n_at_h(h + delta_h_numerical)
        return (n_h_plus - n_h) / delta_h_numerical
    elif h < 0: return 0
    else:
        n_h_plus = get_n_at_h(h + delta_h_numerical)
        n_h_minus = get_n_at_h(h - delta_h_numerical)
        return (n_h_plus - n_h_minus) / (2 * delta_h_numerical)

# --- MODIFIED PLOTTING FUNCTIONS ---
def plot_figure3_temperature_vs_height(h_start_laser, temp_at_ground_C, gradient_type,
                                       lapse_rate_K_per_m, h_limit_max, figure_num="3"):
    heights_plot = np.linspace(0, h_limit_max, 200)
    temperatures_K_profile = [temp_air_profile(h, temp_at_ground_C, gradient_type, lapse_rate_K_per_m) for h in heights_plot]
    temperatures_C_profile = [T - 273.15 for T in temperatures_K_profile]
    temp_at_sea_level_C = temp_air_profile(0, temp_at_ground_C, gradient_type, lapse_rate_K_per_m) - 273.15
    temp_at_h_start_K = temp_air_profile(h_start_laser, temp_at_ground_C, gradient_type, lapse_rate_K_per_m)
    temp_at_h_start_C = temp_at_h_start_K - 273.15

    plt.figure(figsize=(7, 9))
    ax1 = plt.gca()
    ax1.plot(temperatures_C_profile, heights_plot, 'b-', label=f'Temp. Profile ({gradient_type})')
    ax1.set_xlabel('Temperature (°C)')
    ax1.set_ylabel('Height above Ground (m)')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.plot(temp_at_sea_level_C, 0, 'go', markersize=8, label=f'Sea Level (0m): {temp_at_sea_level_C:.2f}°C')
    if 0 <= h_start_laser <= h_limit_max :
        ax1.plot(temp_at_h_start_C, h_start_laser, 'ro', markersize=8, label=f'Laser Start ({h_start_laser}m): {temp_at_h_start_C:.2f}°C')
        ax1.axhline(y=h_start_laser, color='r', linestyle=':', alpha=0.5, label=f'_nolegend_') # No separate legend for line

    ax2 = ax1.twiny()
    ax2.plot([T_K + 273.15 for T_K in temperatures_C_profile], heights_plot, color='None')
    ax2.set_xlabel('Temperature (K)')
    handles1, labels1 = ax1.get_legend_handles_labels()
    ax1.legend(handles1, labels1, loc='best')
    title_parts = [f'Figure {figure_num}: Air Temperature vs. Height',
                   f'Ground (0m) Temp: {temp_at_ground_C}°C']
    if gradient_type == "linear_lapse":
        title_parts.append(f'Lapse Rate: {lapse_rate_K_per_m:.4f} K/m ({"Cooler" if lapse_rate_K_per_m > 0 else "Warmer"} with height)')
    plt.title('\n'.join(title_parts), pad=20)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

def plot_figure4_beam_landing_separation(final_y_positions, initial_angles_mrad, figure_num="4"):
    if not final_y_positions or len(final_y_positions) < 2:
        print(f"Figure {figure_num}: Not enough beam landing positions to calculate separations.")
        return

    separations = []
    pair_labels = []
    for i in range(len(final_y_positions) - 1):
        sep = final_y_positions[i+1] - final_y_positions[i]
        separations.append(sep)
        label = f"B{i+1}-B{i+2}"
        # CORRECTED LINE BELOW:
        if initial_angles_mrad is not None and len(initial_angles_mrad) == len(final_y_positions):
             label += f"\n({initial_angles_mrad[i]:.2f} to {initial_angles_mrad[i+1]:.2f} mrad)"
        pair_labels.append(label)

    if not separations:
        print(f"Figure {figure_num}: No separations calculated.")
        return

    avg_separation = np.mean(separations)
    plt.figure(figsize=(max(10, len(separations)*0.8), 7))
    x_indices = np.arange(len(separations))
    plt.bar(x_indices, separations, color='coral', label='Separation between adjacent beams (m)')
    plt.axhline(avg_separation, color='dodgerblue', linestyle='--', linewidth=2, label=f'Average Separation: {avg_separation:.4f} m')
    plt.xlabel('Adjacent Beam Pair')
    plt.ylabel('Landing Height Separation (m) [Y(beam N+1) - Y(beam N)]')
    plt.title(f'Figure {figure_num}: Difference in Landing Height Between Adjacent Beams')
    if pair_labels:
        plt.xticks(x_indices, pair_labels, rotation=45, ha="right", fontsize=9)
    plt.legend(); plt.grid(True, axis='y', linestyle='--', alpha=0.7); plt.tight_layout()

# --- Main Simulation ---
def run_simulation(
    dist_wall_sim=dist_wall_default,
    n_beams_sim=n_beams_default,
    d_angle_mrad_sim=d_angle_mrad_default,
    custom_initial_angles_rad_sim=None, # NEW: For providing custom initial angles
    w_laser_nm_sim=w_laser_nm_default,
    d_step_ds_sim=d_step_ds_default,
    h_limit_max_sim=h_limit_max_default,
    h_limit_min_sim=h_limit_min_default,
    h_start_sim=h_start_default,
    temp_at_ground_C_sim=temp_at_ground_C_default,
    gradient_type_sim=gradient_type_default,
    custom_lapse_rate_K_per_m=DEFAULT_LAPSE_RATE,
    p0_Pa_sim=P_STD_PA
    ):

    if h_start_sim <= h_limit_min_sim and not (abs(h_start_sim - h_limit_min_sim) < 1e-9) : # allow starting exactly at ground
        print(f"Warning: Laser start height ({h_start_sim}m) is below ground limit ({h_limit_min_sim}m). Adjust parameters.")
    if h_start_sim > h_limit_max_sim:
         print(f"Warning: Laser start height ({h_start_sim}m) is above max height limit ({h_limit_max_sim}m). Adjust parameters.")


    print(f"\n--- Starting Simulation ---")
    print(f"Laser Start Height: {h_start_sim}m")
    # ... (rest of print statements from previous version) ...
    print(f"Wall Dist: {dist_wall_sim}m | Num Beams: {n_beams_sim} | Ang. Sep: {d_angle_mrad_sim}mrad | Lambda: {w_laser_nm_sim}nm")
    print(f"Ground Temp: {temp_at_ground_C_sim}°C | Temp Grad: '{gradient_type_sim}'")

    L_for_pressure_sim = 0.0
    if gradient_type_sim == "linear_lapse":
        L_for_pressure_sim = custom_lapse_rate_K_per_m
        print(f"Temp Profile Lapse Rate: {custom_lapse_rate_K_per_m} K/m "
              f"({'Cooler' if custom_lapse_rate_K_per_m > 0 else ('Warmer' if custom_lapse_rate_K_per_m < 0 else 'Constant')} with height)")
        print(f"Pressure calculation will use lapse rate: {L_for_pressure_sim} K/m")
    else:
        print(f"Pressure calculation will use isothermal model (L_lapse = 0 K/m)")
    print(f"Prop Step: {d_step_ds_sim}m | Height Limits: Ground={h_limit_min_sim}m, Max={h_limit_max_sim}m")
    print(f"Initial Pressure @h=0: {p0_Pa_sim} Pa")


    lambda_vac_m_sim = w_laser_nm_sim * 1e-9
    d_angle_rad_sim = d_angle_mrad_sim * 1e-3

    # Handle initial angles (custom or linspace)
    if custom_initial_angles_rad_sim is not None:
        initial_angles_rad = np.asarray(custom_initial_angles_rad_sim)
        if n_beams_sim != len(initial_angles_rad) and len(initial_angles_rad) > 0 :
            print(f"Warning: n_beams_sim ({n_beams_sim}) does not match length of custom_initial_angles_rad "
                  f"({len(initial_angles_rad)}). Using length of custom angles ({len(initial_angles_rad)}) for n_beams_sim.")
            n_beams_sim = len(initial_angles_rad)
        elif len(initial_angles_rad) == 0: # custom angles is empty, fall back
            print(f"Warning: custom_initial_angles_rad was empty. Falling back to n_beams_sim and d_angle_mrad_sim.")
            initial_angles_rad = np.linspace(
                -(n_beams_sim - 1) * d_angle_rad_sim / 2.0,
                (n_beams_sim - 1) * d_angle_rad_sim / 2.0,
                max(1, n_beams_sim) # ensure at least 1 beam if n_beams_sim was 0
            )
    else:
        if n_beams_sim <=0: # handle n_beams_sim=0 case if no custom angles
            print("Warning: n_beams_sim is 0 or negative. Simulating 1 beam at 0 angle.")
            n_beams_sim = 1
            initial_angles_rad = np.array([0.0])
        else:
            initial_angles_rad = np.linspace(
                -(n_beams_sim - 1) * d_angle_rad_sim / 2.0,
                (n_beams_sim - 1) * d_angle_rad_sim / 2.0,
                n_beams_sim
            )
    initial_angles_mrad = initial_angles_rad * 1000

    all_ray_paths = []
    final_y_positions = []

    for i_beam, phi_initial in enumerate(initial_angles_rad):
        x, y = 0.0, h_start_sim
        phi = phi_initial

        if y > h_limit_max_sim :
            print(f"Beam {i_beam+1} ({initial_angles_mrad[i_beam]:.3f} mrad): Starts ABOVE max height limit. Terminating.")
            all_ray_paths.append(([x], [h_limit_max_sim]))
            final_y_positions.append(h_limit_max_sim)
            continue
        if y < h_limit_min_sim: # Check if starting below ground
            print(f"Beam {i_beam+1} ({initial_angles_mrad[i_beam]:.3f} mrad): Starts AT/BELOW min height limit ({h_limit_min_sim}m). Terminating.")
            all_ray_paths.append(([x], [h_limit_min_sim]))
            final_y_positions.append(h_limit_min_sim)
            continue

        current_path_x = [x]; current_path_y = [y]
        total_s_propagated = 0.0; terminated_early = False

        while x < dist_wall_sim:
            T_curr_K = temp_air_profile(y, temp_at_ground_C_sim, gradient_type_sim, custom_lapse_rate_K_per_m)
            T_ground_K_for_pressure = temp_at_ground_C_sim + 273.15
            P_curr_Pa = pressure_at_height(y, T_ground_K_for_pressure, p0_Pa_sim, L_for_pressure_sim)

            if T_curr_K <= 0 or P_curr_Pa <= 0:
                print(f"Beam {i_beam+1} ({initial_angles_mrad[i_beam]:.3f} mrad): Unphysical T={T_curr_K:.2f}K, P={P_curr_Pa:.2f}Pa at y={y:.2f}m. Stopping.")
                terminated_early = True; break
            n_curr = refractive_index_air(T_curr_K, P_curr_Pa, lambda_vac_m_sim)
            grad_n_y = calculate_dn_dh(y, temp_at_ground_C_sim, p0_Pa_sim, lambda_vac_m_sim,
                                       gradient_type_sim, custom_lapse_rate_K_per_m)
            cos_phi = np.cos(phi); sin_phi = np.sin(phi)
            dphi_ds = (1.0 / n_curr) * grad_n_y * cos_phi
            x += cos_phi * d_step_ds_sim; y += sin_phi * d_step_ds_sim
            phi += dphi_ds * d_step_ds_sim; total_s_propagated += d_step_ds_sim
            current_path_x.append(x); current_path_y.append(y)

            if y > h_limit_max_sim:
                print(f"Beam {i_beam+1} ({initial_angles_mrad[i_beam]:.3f} mrad): Exceeded MAX height ({h_limit_max_sim}m). Terminating.")
                current_path_y[-1] = h_limit_max_sim; terminated_early = True; break
            if y < h_limit_min_sim:
                print(f"Beam {i_beam+1} ({initial_angles_mrad[i_beam]:.3f} mrad): Hit GROUND ({h_limit_min_sim}m). Terminating.")
                current_path_y[-1] = h_limit_min_sim; terminated_early = True; break
            if total_s_propagated > 2 * (dist_wall_sim + h_limit_max_sim + abs(h_limit_min_sim)):
                print(f"Beam {i_beam+1} ({initial_angles_mrad[i_beam]:.3f} mrad): Propagated {total_s_propagated:.1f}m. Stopping.")
                terminated_early = True; break
        
        all_ray_paths.append((current_path_x, current_path_y))
        final_y_at_termination = current_path_y[-1]
        if not terminated_early and x >= dist_wall_sim and len(current_path_x) > 1:
            try:
                final_y_at_termination = np.interp(dist_wall_sim, current_path_x, current_path_y)
                final_y_at_termination = max(h_limit_min_sim, min(h_limit_max_sim, final_y_at_termination))
            except Exception as e:
                print(f"Interpolation failed for beam {i_beam+1}: {e}.")
        final_y_positions.append(final_y_at_termination)
        # (Simplified print message here, full status print at the end)

    # --- Plotting ---
    # Figure 1
    plt.figure(figsize=(12, 8)) # Ray Paths
    for i, (px, py) in enumerate(all_ray_paths):
        plt.plot(px, py, label=f'B{i+1} ({initial_angles_mrad[i]:.2f} mrad)')
    plt.axvline(x=dist_wall_sim, color='k', linestyle='--', label=f'Wall {dist_wall_sim}m')
    plt.axhline(y=h_limit_max_sim, color='r', linestyle=':', label=f'Max H {h_limit_max_sim}m')
    plt.axhline(y=h_limit_min_sim, color='g', linestyle=':', label=f'Ground {h_limit_min_sim}m')
    plt.axhline(y=h_start_sim, color='purple', linestyle='-.', alpha=0.7, label=f'Start H {h_start_sim}m')
    for i, y_final_pos in enumerate(final_y_positions): # Mark impacts
        last_x_of_beam = all_ray_paths[i][0][-1] if all_ray_paths[i][0] else 0
        if last_x_of_beam >= dist_wall_sim - d_step_ds_sim :
            is_early_term_max = (abs(y_final_pos - h_limit_max_sim) < 1e-3 and last_x_of_beam < dist_wall_sim - d_step_ds_sim)
            is_early_term_min = (abs(y_final_pos - h_limit_min_sim) < 1e-3 and last_x_of_beam < dist_wall_sim - d_step_ds_sim)
            if not (is_early_term_max or is_early_term_min):
                plt.plot(dist_wall_sim, y_final_pos, 'ko', markersize=5)
    plt.xlabel('Distance (m)'); plt.ylabel('Height (m)')
    title_parts_fig1 = [f'Figure 1: Laser Beam Propagation ({gradient_type_sim})',
                        f'Ground T: {temp_at_ground_C_sim}°C, Start H: {h_start_sim}m']
    if gradient_type_sim == "linear_lapse": title_parts_fig1.append(f'Lapse: {custom_lapse_rate_K_per_m:.4f} K/m')
    plt.title('\n'.join(title_parts_fig1)); plt.legend(fontsize='small', loc='best'); plt.grid(True, alpha=0.7)
    all_y_vals = [y for _, p_y in all_ray_paths for y in p_y if p_y]
    if all_y_vals:
        min_y_data = min(all_y_vals); max_y_data = max(all_y_vals)
        plot_min_y = min(h_limit_min_sim, min_y_data, h_start_sim) -1
        plot_max_y = max(h_limit_max_sim, max_y_data, h_start_sim) +1
        plt.ylim(max(h_limit_min_sim -2, plot_min_y), min(h_limit_max_sim +2, plot_max_y))
    plt.xlim(0, dist_wall_sim + dist_wall_sim * 0.05)

    # Figure 2
    valid_final_y_at_wall_plot = []
    valid_initial_angles_for_plot = []
    for i, y_final_pos in enumerate(final_y_positions):
        last_x_of_beam = all_ray_paths[i][0][-1] if all_ray_paths[i][0] else 0
        is_at_max_limit_early = (abs(y_final_pos - h_limit_max_sim) < 1e-3 and last_x_of_beam < dist_wall_sim - d_step_ds_sim)
        is_at_min_limit_early = (abs(y_final_pos - h_limit_min_sim) < 1e-3 and last_x_of_beam < dist_wall_sim - d_step_ds_sim)
        if last_x_of_beam >= dist_wall_sim - d_step_ds_sim and not is_at_max_limit_early and not is_at_min_limit_early:
            valid_final_y_at_wall_plot.append(y_final_pos)
            valid_initial_angles_for_plot.append(initial_angles_mrad[i])
    if valid_final_y_at_wall_plot:
        plt.figure(figsize=(8,6))
        plt.plot(valid_initial_angles_for_plot, valid_final_y_at_wall_plot, 'bo-', label="Impact on Wall")
        plt.xlabel("Initial Angle (mrad)"); plt.ylabel(f"Final Height at Wall ({dist_wall_sim}m)")
        plt.title("Figure 2: Beam Landing Height at Wall vs. Initial Angle")
        plt.grid(True, alpha=0.7); plt.legend()

    # Figure 3
    plot_figure3_temperature_vs_height(h_start_sim, temp_at_ground_C_sim, gradient_type_sim,
                                       custom_lapse_rate_K_per_m, h_limit_max_sim, figure_num="3")
    # Figure 4
    plot_figure4_beam_landing_separation(final_y_positions, initial_angles_mrad, figure_num="4")

    plt.show()

    print("\n--- Final Y positions (at termination: wall, ground, or max height) ---")
    for i, y_pos in enumerate(final_y_positions):
        status = ""; last_x_beam = all_ray_paths[i][0][-1] if all_ray_paths[i] and all_ray_paths[i][0] else -1
        if abs(y_pos - h_limit_max_sim) < 1e-3 : status = f"(At Max Height {h_limit_max_sim}m)"
        elif abs(y_pos - h_limit_min_sim) < 1e-3: status = f"(At Ground {h_limit_min_sim}m)"
        elif last_x_beam >= dist_wall_sim - d_step_ds_sim: status = f"(At Wall x~{dist_wall_sim}m)"
        else: status = f"(Terminated early x={last_x_beam:.2f}m)"
        print(f"Beam {i+1} (Angle: {initial_angles_mrad[i]:.3f} mrad): Final Y = {y_pos:.4f} m {status}")
    return all_ray_paths, final_y_positions, initial_angles_mrad

if __name__ == '__main__':
    print("### Running Default Simulation (Standard Lapse, Start Height 2m) ###")
    run_simulation(
        h_start_sim=2.0, temp_at_ground_C_sim=20.0, gradient_type_sim="linear_lapse",
        custom_lapse_rate_K_per_m=0.0065, n_beams_sim=11, d_angle_mrad_sim=0.05,
        dist_wall_sim=1000, h_limit_min_sim=0.0
    )
    print("\n### Running Simulation with Constant Temperature (Start Height 0.5m) ###")
    run_simulation(
        h_start_sim=0.5, temp_at_ground_C_sim=15.0, gradient_type_sim="constant",
        custom_lapse_rate_K_per_m=0.0, n_beams_sim=7, d_angle_mrad_sim=0.1,
        dist_wall_sim=500, h_limit_min_sim=0.0
    )
    print("\n### Running Simulation with Strong Temperature Inversion (Start Height 1m) ###")
    run_simulation(
        h_start_sim=1.0, temp_at_ground_C_sim=10.0, gradient_type_sim="linear_lapse",
        custom_lapse_rate_K_per_m=-0.05, h_limit_max_sim=50, n_beams_sim=9,
        d_angle_mrad_sim=0.08, dist_wall_sim=1500, h_limit_min_sim=0.0
    )
    print("\n### Running Simulation with Custom Angles (Start Height 0.1m) ###")
    run_simulation(
        h_start_sim=0.1, temp_at_ground_C_sim=25.0, gradient_type_sim="linear_lapse",
        custom_lapse_rate_K_per_m=0.0098,
        # Pass custom angles here using the new parameter name
        custom_initial_angles_rad_sim=np.array([-0.0004, -0.0002, 0, 0.0001, 0.0002]),
        n_beams_sim=5, # Should match length of custom_initial_angles_rad_sim, or will be adjusted
        dist_wall_sim=300, h_limit_min_sim=0.0
    )