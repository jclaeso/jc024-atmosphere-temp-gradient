import numpy as np
import matplotlib.pyplot as plt

# --- Physical Constants ---
G_ACC = 9.80665  # Gravitational acceleration (m/s^2)
MOLAR_MASS_AIR = 0.0289644  # Molar mass of dry air (kg/mol)
R_GAS_CONST = 8.31447  # Universal gas constant (J/(mol*K))
P_STD_PA = 101325.0  # Standard pressure (Pa)
T_STD_EDLEN_K = 288.15  # Standard temperature for Edlen's formula (15°C in K)
DEFAULT_LAPSE_RATE = 0.0065 # K/m

# --- Simulation Parameters ---
# User-defined variables with start values
dist_wall = 500.0  # distance from the laser to the wall (m)
n_beams = 10  # number of laser beams
d_angle_mrad = 0.1  # angular separation between beams (mrad)
w_laser_nm = 660.0  # wavelength of the laser (nm)
d_step_ds = 0.1  # step size for path length propagation (m)
h_limit_max = 30.0  # rays above this height are not considered (m)
h_limit_min = -1.0 # rays below this height (e.g. ground) are not considered (m)

# Default atmospheric conditions
temp_at_ground_C_default = 20.0  # Temperature at h=0 in Celsius
gradient_type_default = "linear_lapse" # "constant" or "linear_lapse"

# --- Helper Functions ---

def temp_air_profile(h, temp_at_ground_C=temp_at_ground_C_default,
                     gradient_type="linear_lapse",
                     lapse_rate_K_per_m=DEFAULT_LAPSE_RATE):
    """
    Calculates the air temperature at a given height.

    Args:
        h (float): Height above ground (m).
        temp_at_ground_C (float): Temperature at ground level (h=0) in Celsius.
        gradient_type (str): "constant" or "linear_lapse".
        lapse_rate_K_per_m (float): Temperature lapse rate in K/m for linear_lapse.
                                      Positive value means temperature decreases with height.
    Returns:
        float: Temperature in Kelvin at height h.
    """
    T0_K = temp_at_ground_C + 273.15

    if h < 0: # Should not happen if h_limit_min is effective
        h = 0

    if gradient_type == "constant":
        return T0_K
    elif gradient_type == "linear_lapse":
        T_K_at_h = T0_K - lapse_rate_K_per_m * h
        return T_K_at_h if T_K_at_h > 0 else 1.0 # Avoid non-positive Kelvin temps
    # elif gradient_type == "polynomial":
    #     # Placeholder for polynomial gradient
    #     # params could be [coeff0, coeff1, coeff2] for T(h) = c0 + c1*h + c2*h^2 (in C or K)
    #     pass
    else: # Default to linear lapse if type unknown
        T_K_at_h = T0_K - lapse_rate_K_per_m * h
        return T_K_at_h if T_K_at_h > 0 else 1.0


def pressure_at_height(h, temp_at_ground_K, p0_Pa=P_STD_PA,
                       L_lapse=DEFAULT_LAPSE_RATE):
    """
    Calculates air pressure at height h using the barometric formula for a linear temperature lapse.

    Args:
        h (float): Height above sea level (m).
        temp_at_ground_K (float): Temperature at sea level (h=0) in Kelvin.
        p0_Pa (float): Pressure at sea level (Pa).
        L_lapse (float): Standard temperature lapse rate (K/m).
    Returns:
        float: Pressure in Pascals at height h.
    """
    if h < 0: # Should not happen
        h = 0

    if temp_at_ground_K <= 0: # Invalid T0
        return p0_Pa * np.exp(-G_ACC * MOLAR_MASS_AIR * h / (R_GAS_CONST * 1.0)) # Fallback to isothermal with 1K

    if L_lapse == 0: # Isothermal atmosphere
        return p0_Pa * np.exp(-G_ACC * MOLAR_MASS_AIR * h / (R_GAS_CONST * temp_at_ground_K))

    base = 1.0 - (L_lapse * h) / temp_at_ground_K

    if base <= 0:
        # This implies h is very high, or T0 is too low. Pressure effectively zero or very low.
        # This typically happens above the tropopause where this formula version is not valid
        # or if T0 implies atmosphere ends. For our h_limit, this should not be common.
        return 1e-3 # Return a very small positive pressure

    exponent = (G_ACC * MOLAR_MASS_AIR) / (R_GAS_CONST * L_lapse)
    P_h = p0_Pa * (base ** exponent)
    return P_h if P_h > 0 else 1e-3


def refractive_index_air(T_K, P_Pa, lambda_vac_m):
    """
    Calculates the refractive index of dry air using a simplified Edlen's formula
    and pressure/temperature scaling.

    Args:
        T_K (float): Temperature in Kelvin.
        P_Pa (float): Pressure in Pascals.
        lambda_vac_m (float): Vacuum wavelength of light in meters.
    Returns:
        float: Refractive index of air.
    """
    if T_K <= 0: T_K = 1.0 # Avoid division by zero
    if P_Pa <=0: P_Pa = 1e-3

    lambda_vac_um = lambda_vac_m * 1e6  # Convert wavelength to micrometers
    sigma_sq = (1 / lambda_vac_um)**2

    # Edlen's formula for refractivity (n-1) of standard air (15°C, 101325 Pa)
    # (n_s - 1) * 10^8 = 8342.13 + 2406030 / (130 - sigma^2) + 15997 / (38.9 - sigma^2)
    # More recent coefficients might be slightly different, e.g. Ciddor equation is more complex
    # Using values from Wikipedia / common sources for Edlen (1966 like)
    # Coefficients for (n-1)s * 10^8
    A = 8342.54
    B = 2406147
    C = 130
    D = 15998
    E = 38.9

    # Avoid division by zero if wavelength matches poles (unlikely for visible light)
    if abs(C - sigma_sq) < 1e-6 or abs(E - sigma_sq) < 1e-6: # highly unlikely
        # Fallback or error, this means wavelength is resonant with absorption lines
        # For this simulation, we can use a nearby value or average n_air ~1.00027
        n_s_minus_1_e8 = 8342.54 + 2406147 / (130 - (1/0.55)**2) + 15998 / (38.9 - (1/0.55)**2) # at 550nm
    else:
        n_s_minus_1_e8 = A + B / (C - sigma_sq) + D / (E - sigma_sq)

    n_s_minus_1 = n_s_minus_1_e8 / 1e8

    # Adjust for actual temperature (T_K) and pressure (P_Pa)
    # (n - 1) = (n_s - 1) * (P_Pa / P_STD_PA) * (T_STD_EDLEN_K / T_K) (Ideal gas density scaling)
    # This is a common simplification. More precise formulas exist (e.g. involving compressibility)
    n_minus_1 = n_s_minus_1 * (P_Pa / P_STD_PA) * (T_STD_EDLEN_K / T_K)

    return 1.0 + n_minus_1


def calculate_dn_dh(h, temp_at_ground_C, p0_Pa, lambda_vac_m,
                    gradient_type, lapse_rate_K_per_m,
                    delta_h_numerical=0.01):
    """
    Calculates the gradient of refractive index dn/dh numerically.

    Args:
        h (float): Current height (m).
        temp_at_ground_C (float): Ground temperature in Celsius.
        p0_Pa (float): Ground pressure in Pascals.
        lambda_vac_m (float): Vacuum wavelength in meters.
        gradient_type (str): Temperature gradient type.
        lapse_rate_K_per_m (float): Lapse rate for linear gradient.
        delta_h_numerical (float): Small step for numerical derivative.
    Returns:
        float: dn/dh.
    """
    T_ground_K = temp_at_ground_C + 273.15

    # Temperature profile function arguments
    temp_args = {
        "temp_at_ground_C": temp_at_ground_C,
        "gradient_type": gradient_type,
        "lapse_rate_K_per_m": lapse_rate_K_per_m
    }

    # Pressure function arguments for P(h) calculation
    # Note: pressure_at_height uses its own lapse rate parameter for the barometric formula model,
    # which is typically the standard atmospheric lapse rate (DEFAULT_LAPSE_RATE).
    # The temp_at_ground_K for pressure function is the actual ground temp.
    pressure_args = {
        "temp_at_ground_K": T_ground_K,
        "p0_Pa": p0_Pa,
        "L_lapse": DEFAULT_LAPSE_RATE # Standard lapse for pressure model
    }

    if h < delta_h_numerical and h >= 0: # Use forward difference near ground
        T_h = temp_air_profile(h, **temp_args)
        P_h = pressure_at_height(h, **pressure_args)
        n_h = refractive_index_air(T_h, P_h, lambda_vac_m)

        T_h_plus = temp_air_profile(h + delta_h_numerical, **temp_args)
        P_h_plus = pressure_at_height(h + delta_h_numerical, **pressure_args)
        n_h_plus = refractive_index_air(T_h_plus, P_h_plus, lambda_vac_m)

        return (n_h_plus - n_h) / delta_h_numerical
    elif h < 0: # Should be caught by h_limit_min, but as a fallback
        return 0 # No gradient below ground

    else: # Use central difference
        h_plus = h + delta_h_numerical
        h_minus = h - delta_h_numerical

        T_h_plus = temp_air_profile(h_plus, **temp_args)
        P_h_plus = pressure_at_height(h_plus, **pressure_args)
        n_h_plus = refractive_index_air(T_h_plus, P_h_plus, lambda_vac_m)

        T_h_minus = temp_air_profile(h_minus, **temp_args)
        P_h_minus = pressure_at_height(h_minus, **pressure_args)
        n_h_minus = refractive_index_air(T_h_minus, P_h_minus, lambda_vac_m)

        return (n_h_plus - n_h_minus) / (2 * delta_h_numerical)


# --- Main Simulation ---
def run_simulation(
    dist_wall_sim=dist_wall,
    n_beams_sim=n_beams,
    d_angle_mrad_sim=d_angle_mrad,
    w_laser_nm_sim=w_laser_nm,
    d_step_ds_sim=d_step_ds,
    h_limit_max_sim=h_limit_max,
    h_limit_min_sim=h_limit_min,
    temp_at_ground_C_sim=temp_at_ground_C_default,
    gradient_type_sim=gradient_type_default,
    custom_lapse_rate_K_per_m=DEFAULT_LAPSE_RATE, # This is for the temp_air_profile if linear
    p0_Pa_sim=P_STD_PA,
    start_height_m = 0.001 # Start slightly above ground to help numerics for dn/dh
    ):

    print(f"Starting simulation with: Wall Distance={dist_wall_sim}m, Num Beams={n_beams_sim}, "
          f"Ang. Sep.={d_angle_mrad_sim}mrad, Lambda={w_laser_nm_sim}nm")
    print(f"Ground Temp={temp_at_ground_C_sim}°C, Temp. Grad. Type='{gradient_type_sim}'")
    if gradient_type_sim == "linear_lapse":
        print(f"Custom Lapse Rate for Temp Profile={custom_lapse_rate_K_per_m} K/m (Std Pressure Lapse Rate={DEFAULT_LAPSE_RATE} K/m)")
    print(f"Propagation Step={d_step_ds_sim}m, Height Limit Max={h_limit_max_sim}m, Min={h_limit_min_sim}m")
    print(f"Initial Pressure at h=0: {p0_Pa_sim} Pa")


    lambda_vac_m_sim = w_laser_nm_sim * 1e-9
    d_angle_rad_sim = d_angle_mrad_sim * 1e-3

    initial_angles = np.linspace(
        -(n_beams_sim - 1) * d_angle_rad_sim / 2.0,
         (n_beams_sim - 1) * d_angle_rad_sim / 2.0,
        n_beams_sim
    )

    all_ray_paths = []
    final_y_positions = []

    for i_beam, phi_initial in enumerate(initial_angles):
        x, y = 0.0, start_height_m
        phi = phi_initial

        current_path_x = [x]
        current_path_y = [y]

        total_s_propagated = 0.0

        # Propagate ray step by step using path length `ds`
        # Stop when x exceeds dist_wall_sim or other limits are hit
        while x < dist_wall_sim:
            # Get current atmospheric conditions at height y
            T_curr_K = temp_air_profile(y, temp_at_ground_C_sim, gradient_type_sim, custom_lapse_rate_K_per_m)
            P_curr_Pa = pressure_at_height(y, temp_at_ground_C_sim + 273.15, p0_Pa_sim, DEFAULT_LAPSE_RATE)

            if T_curr_K <= 0 or P_curr_Pa <= 0: # Safety break
                print(f"Beam {i_beam+1}: Unphysical atmospheric condition T={T_curr_K}K, P={P_curr_Pa}Pa at y={y}m. Stopping ray.")
                break

            n_curr = refractive_index_air(T_curr_K, P_curr_Pa, lambda_vac_m_sim)

            grad_n_y = calculate_dn_dh(y, temp_at_ground_C_sim, p0_Pa_sim, lambda_vac_m_sim,
                                       gradient_type_sim, custom_lapse_rate_K_per_m)

            # Ray equations (Euler integration step)
            # dx/ds = cos(phi)
            # dy/ds = sin(phi)
            # dphi/ds = (1/n) * (dn/dy) * cos(phi)

            cos_phi = np.cos(phi)
            sin_phi = np.sin(phi)

            dx_ds = cos_phi
            dy_ds = sin_phi
            dphi_ds = (1.0 / n_curr) * grad_n_y * cos_phi

            # Update positions and angle
            x += dx_ds * d_step_ds_sim
            y += dy_ds * d_step_ds_sim
            phi += dphi_ds * d_step_ds_sim

            total_s_propagated += d_step_ds_sim

            current_path_x.append(x)
            current_path_y.append(y)

            # Check limits
            if y > h_limit_max_sim:
                print(f"Beam {i_beam+1}: Exceeded max height limit ({h_limit_max_sim} m) at x={x:.2f} m, y={y:.2f} m.")
                break
            if y < h_limit_min_sim: # Hit ground
                print(f"Beam {i_beam+1}: Hit ground limit ({h_limit_min_sim} m) at x={x:.2f} m, y={y:.2f} m.")
                # Interpolate to find y at ground if needed, or just take last y.
                # For simplicity, we stop and record the last point.
                # If it matters, one could adjust the last step to hit y=0 exactly.
                y = h_limit_min_sim
                current_path_y[-1] = y
                break

            if total_s_propagated > 2 * dist_wall_sim : # Safety break if ray turns back significantly
                print(f"Beam {i_beam+1}: Propagated too far ({total_s_propagated:.2f}m) without reaching wall. Stopping.")
                break

        all_ray_paths.append((current_path_x, current_path_y))

        # Interpolate to find y at x = dist_wall_sim if the ray passed it
        final_y = y
        if x >= dist_wall_sim and len(current_path_x) > 1:
            # Find the point where x crosses dist_wall_sim
            try:
                final_y = np.interp(dist_wall_sim, current_path_x, current_path_y)
            except Exception as e:
                print(f"Interpolation failed for beam {i_beam+1}: {e}. Using last y.")
                final_y = current_path_y[-1] if current_path_y else 0
        elif y > h_limit_max_sim:
            final_y = h_limit_max_sim # Or mark as 'above limit'
        elif y < h_limit_min_sim:
            final_y = h_limit_min_sim # Or mark as 'hit ground'

        final_y_positions.append(final_y)
        print(f"Beam {i_beam+1}: Initial Angle={phi_initial*1000:.3f} mrad -> Final X={x:.2f}m, Final Y (at wall or limit)={final_y:.3f}m")

    # --- Plotting ---
    plt.figure(figsize=(12, 8))
    for i, (px, py) in enumerate(all_ray_paths):
        plt.plot(px, py, label=f'Beam {i+1} ({initial_angles[i]*1000:.2f} mrad)')

    plt.axvline(x=dist_wall_sim, color='k', linestyle='--', label=f'Wall at {dist_wall_sim}m')
    plt.axhline(y=h_limit_max_sim, color='r', linestyle=':', label=f'Max Height {h_limit_max_sim}m')
    plt.axhline(y=h_limit_min_sim, color='g', linestyle=':', label=f'Min Height (Ground) {h_limit_min_sim}m')

    # Plot final positions on the wall more clearly
    valid_final_y = []
    valid_initial_angles_for_plot = []
    for i, y_final in enumerate(final_y_positions):
        # Only plot if the beam actually reached the wall vicinity or was stopped by limits
        # Check if the last x for this beam was close to or past the wall
        last_x_of_beam = all_ray_paths[i][0][-1] if all_ray_paths[i][0] else 0
        if last_x_of_beam >= dist_wall_sim or y_final == h_limit_max_sim or y_final == h_limit_min_sim :
             if y_final <= h_limit_max_sim and y_final >= h_limit_min_sim and last_x_of_beam >= dist_wall_sim : # only mark if on wall and within bounds
                plt.plot(dist_wall_sim, y_final, 'ko', markersize=5)
             valid_final_y.append(y_final)
             valid_initial_angles_for_plot.append(initial_angles[i])


    plt.xlabel('Distance from Source (m)')
    plt.ylabel('Height above Ground (m)')
    title_parts = [
        f'Laser Beam Propagation ({gradient_type_sim} temp profile)',
        f'Ground Temp: {temp_at_ground_C_sim}°C'
    ]
    if gradient_type_sim == "linear_lapse":
        title_parts.append(f'Temp Lapse: {custom_lapse_rate_K_per_m} K/m')
    plt.title('\n'.join(title_parts))
    plt.legend(fontsize='small', loc='upper left')
    plt.grid(True)
    # Adjust ylim to ensure visibility of important features like ground and relevant beam paths
    min_y_data = min(py for _, py_list in all_ray_paths if py_list for py in py_list) if any(py_list for _, py_list in all_ray_paths if py_list) else h_limit_min_sim
    max_y_data = max(py for _, py_list in all_ray_paths if py_list for py in py_list) if any(py_list for _, py_list in all_ray_paths if py_list) else h_limit_max_sim

    plot_min_y = min(h_limit_min_sim -1, min_y_data -1)
    plot_max_y = max(h_limit_max_sim +1, max_y_data +1)

    # Ensure the plot doesn't become excessively large if beams go very high/low before limits
    plot_min_y = max(plot_min_y, h_limit_min_sim - 5) # Don't show too much below ground
    plot_max_y = min(plot_max_y, h_limit_max_sim + 10) # Don't show too much above typical max height if limited

    # If all data is within a smaller range than limits, zoom in a bit more
    if max_y_data < h_limit_max_sim: plot_max_y = min(plot_max_y, max_y_data + 2)
    if min_y_data > h_limit_min_sim: plot_min_y = max(plot_min_y, min_y_data -2)


    plt.ylim(plot_min_y, plot_max_y)
    plt.xlim(0, dist_wall_sim + dist_wall_sim*0.1) # Show a bit beyond the wall

    # Secondary plot: final Y vs initial angle
    if valid_final_y:
        plt.figure(figsize=(8,6))
        plt.plot(np.array(valid_initial_angles_for_plot)*1000, valid_final_y, 'bo-')
        plt.xlabel("Initial Angle (mrad)")
        plt.ylabel(f"Final Height at x={dist_wall_sim}m (or limit)")
        plt.title("Beam Landing Height vs. Initial Angle")
        plt.grid(True)

    plt.show()

    print("\nFinal Y positions at wall or limit:")
    for i, y_pos in enumerate(final_y_positions):
        status = ""
        last_x_beam = all_ray_paths[i][0][-1] if all_ray_paths[i][0] else 0
        if y_pos >= h_limit_max_sim - 1e-3 : status = "(At Max Height Limit)"
        elif y_pos <= h_limit_min_sim + 1e-3: status = "(At Min Height Limit/Ground)"
        elif not (last_x_beam >= dist_wall_sim - d_step_ds_sim*2) : status = "(Did not reach wall)"
        print(f"Beam {i+1} (Angle: {initial_angles[i]*1000:.3f} mrad): Y = {y_pos:.4f} m {status}")

    return all_ray_paths, final_y_positions

if __name__ == '__main__':
    print("Running default simulation...")
    # Standard atmosphere, linear lapse for temperature and pressure model
    run_simulation(
        temp_at_ground_C_sim=20.0,
        gradient_type_sim="linear_lapse",
        custom_lapse_rate_K_per_m=DEFAULT_LAPSE_RATE # Temp profile uses standard lapse
    )

    print("\nRunning simulation with constant temperature profile (20 C)...")
    # Constant temperature profile for n, but pressure still uses standard lapse model from T0=20C
    run_simulation(
        temp_at_ground_C_sim=20.0,
        gradient_type_sim="constant", # Actual T(h) is constant for n calculation
        custom_lapse_rate_K_per_m=0.0 # This is ignored for "constant" but set for clarity
    )

    print("\nRunning simulation with strong temperature inversion (linear lapse for temp profile)...")
    # Example: temperature increases with height (inversion) for the temp_air_profile
    # custom_lapse_rate is negative for inversion
    run_simulation(
        temp_at_ground_C_sim=10.0,
        gradient_type_sim="linear_lapse",
        custom_lapse_rate_K_per_m=-0.01, # Temp increases by 0.01 K/m
        h_limit_max_sim=50 # Increase height limit to see effect
    )

    print("\nRunning simulation with higher ground temperature...")
    run_simulation(
        temp_at_ground_C_sim=35.0,
        gradient_type_sim="linear_lapse",
        custom_lapse_rate_K_per_m=DEFAULT_LAPSE_RATE
    )

    # Example of how to change other parameters:
    # run_simulation(dist_wall_sim=1000, n_beams_sim=5, d_angle_mrad_sim=0.5)