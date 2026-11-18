
#* sim_model/engine_datapath_fixed_point.py

#* ======= Libraries =======
# ------- Natives. -------
from dataclasses import dataclass

#* ======= Config Engine Datapath Fixed Point Dataclass =======
# ------- Custom Type Hints. -------
quantized_values_type = tuple[int, int, int]

@dataclass(frozen = True, slots = True)
class ConfigEngineDatapathFixedPoint:
    #TODO AD
    # ------- Data Representation. -------
    fractional_bit_width: int

    # ------- Ambient Conditions. -------
    ambient_pressure_quantized: int
    ambient_temperature_quantized: int

    # ------- Pre-Calculated & Quantized Time Constants (Inertia). -------
    pre_calculated_rotor_inertial_time_const_quantized: int
    pre_calculated_pressure_volume_time_const_quantized: int
    pre_calculated_thermal_time_const_quantized: int

#* ======= Engine Datapath Fixed Point Class =======
class EngineDatapathFixedPoint:
    #TODO AD
    def __init__(
            self,
            config_engine_datapath_fixed_point: ConfigEngineDatapathFixedPoint
    ) -> None:
        #TODO AD
        self._cfg: ConfigEngineDatapathFixedPoint = config_engine_datapath_fixed_point

    def calculate_target_values(
            self,
            current_fuel_flow: int,
            current_rotational_speed: int,
            gain_constants: tuple[int, int, int, int]
    ) -> quantized_values_type:
        """
        Calculates the quantized target values for engine states using fixed-point arithmetic.

        Mathematical Formulations:
        1. Target Rotational Speed:
        N_target = (K_N * W_f) >> fractional_bit_width
        2. Target Compressor Pressure:
        P_target = P_amb + ((K_P * ((N^2) >> fractional_bit_width)) >> fractional_bit_width)
        3. Target Exhaust Gas Temperature:
        T_target = T_amb + ((K_heat * W_f) >> fractional_bit_width) - ((K_cool * N) >> fractional_bit_width)

        Args:
            current_fuel_flow (int): Current quantized commanded fuel flow rate (W_f) [LSBs].
            current_rotational_speed (int): Current quantized rotational speed (N) [LSBs].
            gain_constants (tuple[int, int, int, int]): Tuple containing quantized system gains:
                - speed_gain_coeff (int): K_N speed gain coefficient.
                - compressor_pressure_gain (int): K_P compressor pressure gain.
                - combustion_heat_gain (int): K_heat combustion thermal gain.
                - mass_overflow_cooling_gain (int): K_cool airflow cooling gain.

        Returns:
            quantized_values_type: Tuple containing target state values in raw LSBs:
                - target_rotational_speed (int): Target speed (N_target).
                - target_compressor_pressure (int): Target pressure (P_target).
                - target_exhaust_gas_temperature (int): Target temperature (T_target).
        """
        # Unpacks the gain constants.
        speed_gain_coefficient, compressor_pressure_gain, combustion_heat_gain, mass_overflow_cooling_gain = gain_constants

        f_width: int = self._cfg.fractional_bit_width

        # ------- Calculates target values. -------
        # All values has to be shifted to the right 'fractional_bit_width' in order to match the original bit widths.
        # Rotational Speed.
        target_rotational_speed: int = (speed_gain_coefficient * current_fuel_flow) >> f_width

        # Compressor Pressure.
        target_compressor_pressure: int = self._cfg.ambient_pressure_quantized + ((compressor_pressure_gain * (((current_rotational_speed)**2) >> f_width)) >> f_width)

        # Exhaust Gas Temperature.
        target_exhaust_gas_temperature: int = self._cfg.ambient_temperature_quantized + ((combustion_heat_gain * current_fuel_flow) >> f_width) - ((mass_overflow_cooling_gain * current_rotational_speed) >> f_width)

        return (target_rotational_speed, target_compressor_pressure, target_exhaust_gas_temperature)

    def calculate_derivatives(
            self,
            current_values: quantized_values_type,
            target_values: quantized_values_type
    ) -> quantized_values_type:
        """
        Calculates the intermediate discrete state changes using division-free scaling.

        Multiplies state error deltas (target - current) by pre-calculated discrete integration
        factors round((dt / tau) * 2^fractional_bit_width), followed by an arithmetic right-shift
        to eliminate hardware dividers and preserve dynamic range.

        Mathematical Formulations:
        1. dN = (C_N * (N_target - N)) >> fractional_bit_width
        2. dP = (C_P * (P_target - P)) >> fractional_bit_width
        3. dT = (C_T * (T_target - T)) >> fractional_bit_width

        Args:
            current_values (quantized_values_type): Tuple of current engine states (N, P, T) in LSBs.
            target_values (quantized_values_type): Tuple of target steady-state values (N_target, P_target, T_target) in LSBs.

        Returns:
            quantized_values_type: Tuple of discrete Euler increment deltas in raw LSBs:
                - d_rotational_speed (int): Discrete step delta for rotational speed.
                - d_compressor_pressure (int): Discrete step delta for compressor pressure.
                - d_exhaust_gas_temperature (int): Discrete step delta for exhaust gas temperature.
        """
        # Unpacks the parameters.
        current_rotational_speed, current_compressor_pressure, current_exhaust_gas_temperature = current_values
        target_rotational_speed, target_compressor_pressure, target_exhaust_gas_temperature = target_values
        # ------- Calculates the derivatives. -------
        # Rotational Speed.
        d_rotational_speed_32b: int = self._cfg.pre_calculated_rotor_inertial_time_const_quantized * (target_rotational_speed - current_rotational_speed)

        # Compressor Pressure.
        d_compressor_pressure_32b: int = self._cfg.pre_calculated_pressure_volume_time_const_quantized * (target_compressor_pressure - current_compressor_pressure)

        # Exhaust Gas Temperature.
        d_exhaust_gas_temperature_32b: int = self._cfg.pre_calculated_thermal_time_const_quantized * (target_exhaust_gas_temperature - current_exhaust_gas_temperature)

        # Shifts the derivativers.
        d_rotational_speed: int = d_rotational_speed_32b >> self._cfg.fractional_bit_width
        d_compressor_pressure: int = d_compressor_pressure_32b >> self._cfg.fractional_bit_width
        d_exhaust_gas_temperature: int = d_exhaust_gas_temperature_32b >> self._cfg.fractional_bit_width

        return (d_rotational_speed, d_compressor_pressure, d_exhaust_gas_temperature)

    def calculate_next_step(
            self,
            current_values: quantized_values_type,
            derivative_values: quantized_values_type
    ) -> quantized_values_type:
        #TODO AD
        # Unpacks the parameters.
        current_rotational_speed, current_compressor_pressure, current_exhaust_gas_temperature = current_values
        d_rotational_speed, d_compressor_pressure, d_exhaust_gas_temperature = derivative_values

        # Uses the Euler equation to compute the values for the next step.
        next_rotational_speed: int = current_rotational_speed + d_rotational_speed

        next_compressor_pressure: int = current_compressor_pressure + d_compressor_pressure

        next_exhaust_gas_temperature: int = current_exhaust_gas_temperature + d_exhaust_gas_temperature

        # Clamps the results to minimum safeguards.
        next_rotational_speed = max(next_rotational_speed, 0)
        next_compressor_pressure = max(next_compressor_pressure, self._cfg.ambient_pressure_quantized)
        next_exhaust_gas_temperature = max(next_exhaust_gas_temperature, self._cfg.ambient_temperature_quantized)

        return (next_rotational_speed, next_compressor_pressure, next_exhaust_gas_temperature)