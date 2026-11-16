
#* sim_model/aircraft_engine_simulator_fixed_point.py

#? In order to calculate how the engine changes over time in fixed-point arithmetic,
#?   these discrete Q16.16 equations must be evaluated at every time step 'k'.
#? Multiplications of two Q16.16 fixed-point values produce a Q32.32 intermediate
#?   result and require an arithmetic right-shift (>> FRACTIONAL_BIT_WIDTH) to realign.
#? Expensive hardware divisions are avoided by pre-calculating discrete (dt / tau) constants.

#* A) Rotational Speed 'N': Engine speed (normalized from 0% to 100% in Q16.16).
# 1) Target Speed: N_target[k] = (K_N * W_f[k]) >> 16.
#   where as: K_N: Speed Gain Coefficient & W_f: Fuel Flow Rate.
# 2) Discrete Derivative (Division-Free): dN[k] = (C_N * (N_target[k] - N[k])) >> 16.
#   C_N: Pre-calculated discrete factor round((dt / tau_N) * 2^16).
# 3) Next State (Euler Accumulation): N[k+1] = N[k] + dN[k].
# 4) Physical Safeguard Clamping: N[k+1] = max(0, N[k+1]).

#* B) Compressor Pressure 'P': Pressure inside the engine (in kPa in Q16.16).
# 1) Target Pressure: P_target[k] = P_amb + ((K_P * ((N[k]^2) >> 16)) >> 16).
#   P_amb: Ambient Atmospheric Pressure & K_P: Compressor Pressure Gain.
# 2) Discrete Derivative (Division-Free): dP[k] = (C_P * (P_target[k] - P[k])) >> 16.
#   C_P: Pre-calculated discrete factor round((dt / tau_P) * 2^16).
# 3) Next State (Euler Accumulation): P[k+1] = P[k] + dP[k].
# 4) Physical Safeguard Clamping: P[k+1] = max(P_amb, P[k+1]).

#* C) Exhaust Gas Temperature 'T': Thermal output (in *C in Q16.16).
# 1) Target Temperature: T_target[k] = T_amb + ((K_heat * W_f[k]) >> 16) - ((K_cool * N[k]) >> 16).
#   T_amb: Ambient Air Temperature & K_heat: Combustion Heat Gain & K_cool: Airflow Cooling Gain.
# 2) Discrete Derivative (Division-Free): dT[k] = (C_T * (T_target[k] - T[k])) >> 16.
#   C_T: Pre-calculated discrete factor round((dt / tau_T) * 2^16).
# 3) Next State (Euler Accumulation): T[k+1] = T[k] + dT[k].
# 4) Physical Safeguard Clamping: T[k+1] = max(T_amb, T[k+1]).

#? Fixed-Point Scaling & Quantization Parameters (Scale Factor = 2^16 = 65536):

# 1) Word Length: 32 bits (16-bit integer, 16-bit fractional, two's-complement signed).
# 2) Simulation Step: dt = 1 ms (0.001 s).
# 3) Ambient Conditions (Quantized):
#   i) P_amb = 101.3 kPa -> round(101.3 * 65536) = 6638797 LSBs.
#   ii) T_amb = 15.0 *C   -> round(15.0 * 65536)  = 983040 LSBs.
# 4) Pre-Calculated Discrete Integration Factors C = round((dt / tau) * 65536):
#   i) C_N = round((0.001 / 1.5) * 65536)  = 44 LSBs (Rotor Inertial Lag).
#   ii) C_P = round((0.001 / 0.15) * 65536) = 437 LSBs (Compressor Volume Lag).
#   iii) C_T = round((0.001 / 0.8) * 65536)  = 82 LSBs (Thermal Mass Lag).
# 5) System Gains (Quantized):
#   i) K_N = 1.0        -> round(1.0 * 65536)     = 65536 LSBs.
#   ii) K_P = 0.015      -> round(0.015 * 65536)   = 983 LSBs.
#   iii) K_heat = 15.0   -> round(15.0 * 65536)    = 983040 LSBs.
#   iv) K_cool = 6.0     -> round(6.0 * 65536)     = 393216 LSBs.
# 6) Structural Safety Trip Thresholds:
#   i) N_max = 110.0%    -> round(110.0 * 65536)   = 7208960 LSBs (Overspeed).
#   ii) T_max = 1100.0 *C -> round(1100.0 * 65536) = 72089600 LSBs (Thermal Meltdown).

#* ======= Libraries =======
# ------- Natives. -------
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

# ------- Externals. -------
import numpy as np

from numpy import float64, int64
from numpy.typing import NDArray
from pandas import DataFrame

# ------- Customs. -------
from sim_model.engine_datapath_fixed_point import EngineDatapathFixedPoint
from sim_model.quantize_floats import quantize_floats

#* ======= Aircraft Engine Simulator's (Fixed Point Type) Config Dataclass =======
@dataclass(frozen = True)
class ConfigAircraftEngineSimulatorFixedPoint:
    """
    Configuration parameters for the discrete fixed-point engine simulator.

    All physical floating-point quantities are pre-scaled and quantized into raw signed integers
    based on the specified integer and fractional bit widths. Runtime divisions are eliminated by
    pre-calculating the discrete Euler integration factors (dt / tau).

    Attributes:
        int_bit_width (int): Bit width allocated for the integer portion of the representation (including sign bit).
        fractional_bit_width (int): Bit width allocated for the fractional portion.
        simulation_step (float): Discrete integration time step (dt) used in the Euler solver [s].
        total_simulation_time (float): Total physical duration of the simulation run [s].
        ambient_pressure_quantized (int): Quantized baseline atmospheric air pressure (P_amb) [Raw LSBs].
        ambient_temperature_quantized (int): Quantized baseline atmospheric air temperature (T_amb) [Raw LSBs].
        speed_gain_coeff_quantized (int): Quantized speed gain coefficient (K_N) mapping fuel to target speed [Raw LSBs].
        compressor_pressure_gain_quantized (int): Quantized pressure gain coefficient (K_P) mapping speed squared to pressure [Raw LSBs].
        combustion_heat_gain_quantized (int): Quantized combustion thermal coefficient (K_heat) mapping fuel to heat [Raw LSBs].
        mass_overflow_cooling_gain_quantized (int): Quantized airflow cooling coefficient (K_cool) mapping speed to cooling [Raw LSBs].
        max_rotational_speed_quantized (int): Quantized safety limit for engine overspeed failures [Raw LSBs].
        max_compressor_temperature_quantized (int): Quantized safety limit for engine thermal meltdown failures [Raw LSBs].
        software_results_file_path (Path): Filesystem path directory used to export reference verification datasets.
    """
    # ------- Data Representation. -------
    int_bit_width: int
    fractional_bit_width: int

    # ------- Simulation. -------
    simulation_step: float
    total_simulation_time: float

    # ------- Ambient Conditions. -------
    ambient_pressure_quantized: int
    ambient_temperature_quantized: int

    # ------- Gains. -------
    speed_gain_coeff_quantized: int
    compressor_pressure_gain_quantized: int
    combustion_heat_gain_quantized: int
    mass_overflow_cooling_gain_quantized: int

    # ------- Engine Limits. -------
    max_rotational_speed_quantized: int
    max_compressor_temperature_quantized: int

    # ------- Path. -------
    software_results_file_path: Path

#* ======= Aircraft Engine Simulator =======
quantized_values_type = tuple[int, int, int]
histories_type = tuple[
    NDArray[int64],
    NDArray[int64],
    NDArray[int64],
    NDArray[int64],
    NDArray[int64],
    NDArray[float64]
]

class AircraftEngineSimulatorFixedPoint:
    """
    Discrete bit-true fixed-point simulator for a single-spool turbojet engine.

    Implements the engine's thermodynamic and mechanical differential equations using
    Q16.16 two's-complement integer arithmetic and first-order forward Euler numerical
    integration. Runtime division operations are eliminated by pre-calculating scaled
    discrete integration coefficients (dt / tau). This model acts as the golden software
    verification reference for bit-for-bit RTL co-simulation.
    """
    def __init__(
            self,
            config_aircraft_engine_simulator_fixed_point: ConfigAircraftEngineSimulatorFixedPoint,
            engine_datapath_fixed_point: EngineDatapathFixedPoint
    ) -> None:
        #TODO AD
        self._cfg: ConfigAircraftEngineSimulatorFixedPoint = config_aircraft_engine_simulator_fixed_point
        self._engine_datapath_fixed_point: EngineDatapathFixedPoint = engine_datapath_fixed_point

    def _fetch_current_fuel_flow(
            self,
            k: float
    ) -> int:
        """
        Evaluates and quantizes the scheduled fuel flow rate based on simulation time.

        Schedules an operational mission throttle profile:
        - Idle: 10 kg/h for t < 2.0 s
        - Throttle Slam (Rapid Acceleration): 90 kg/h for 2.0 s <= t < 18.0 s
        - Throttle Chop (Deceleration): 10 kg/h for t >= 18.0 s

        The physical fuel flow rate is quantized into a signed integer scaled by 2^fractional_bit_width.

        Args:
            k (float): Current physical simulation time in seconds.

        Returns:
            int: Quantized commanded fuel flow rate (W_f) in raw fixed-point LSBs.
        """
        fuel_flow_float: float

        # Idle fuel flow.
        if k < 2.0:
            fuel_flow_float = 10.0
        # Throttle slam / rapic acceleration.
        elif k < 18.0:
            fuel_flow_float = 90.0
        # Throttle chop / deceleration.
        else:
            fuel_flow_float = 10.0

        # Quantizes the fuel flow.
        return quantize_floats(
            val = fuel_flow_float,
            int_bit_width = self._cfg.int_bit_width,
            fractional_bit_width = self._cfg.fractional_bit_width,
            signed = True
        )

    def _export_histories(
            self,
            histories: histories_type
    ) -> None:
        """
        Exports complete quantized telemetry and state histories to a reference CSV dataset.

        Constructs a structured pandas DataFrame and writes out the golden software verification
        trace used for cycle-accurate, bit-for-bit hardware assertions during Cocotb verification.

        Args:
            histories (histories_type): Tuple of simulation history arrays containing:
                - fuel_flow_history (NDArray[int64]): Commanded fuel flow over time [LSBs].
                - speed_history (NDArray[int64]): Rotational speed history [LSBs].
                - pressure_history (NDArray[int64]): Compressor pressure history [LSBs].
                - temperature_history (NDArray[int64]): Exhaust gas temperature history [LSBs].
                - engine_status_history (NDArray[int64]): Emergency trip flags (1 = tripped, 0 = operational).
                - time_history (NDArray[float64]): Physical simulation time points in seconds.

        Returns:
            None
        """
        # Unpacks the histories.
        fuel_flow_history, speed_history, pressure_history, temperature_history, engine_status_history, time_history = histories

        # Declares a pandas dataframe.
        df = DataFrame(
            data = {
                "fuel_flow": fuel_flow_history,
                "speed": speed_history,
                "pressure": pressure_history,
                "temperature": temperature_history,
                "is_engine_broken": engine_status_history,
                "time": time_history
            }
        )

        # Exports the dataframe as a csv file.
        df.to_csv(
            self._cfg.software_results_file_path,
            index = False
        )

    def _create_time_history(
            self,
            total_steps: int
    ) -> NDArray[float64]:
        #TODO AD
        # Calculates to how many digits the time history should be rounded to.
        simulation_step_str = str(self._cfg.simulation_step)
        #? Decimal(): Class for exact base-10 arithmetic, rather than base-2 binary math.
        simulation_step_decimal = Decimal(value = simulation_step_str)
        #? .as_tuple(): Decomposes a Decimal object to its mathematical parts.
        #   f.e. '0.001' -> DecimalTuple(sign = 0, digits = (1,), exponent = -3)
        raw_exponent: int | str = simulation_step_decimal.as_tuple().exponent #TODO AC
        #?  .exponent: Extracts just the exponent integer of a DecimalTuple.
        simulation_step_decimal_exponent: int = raw_exponent if isinstance(raw_exponent, int) else 0 #TODO AC

        time_history_rounding_decimals: int = max(0, -simulation_step_decimal_exponent)

        # Populates the time history as it is deterministic.
        time_history: NDArray[float64] = np.linspace(
            start = 0.0,
            stop = self._cfg.total_simulation_time,
            num = total_steps + 1,
            dtype = float64
        ).round(decimals = time_history_rounding_decimals)

        return time_history

    def _initialize_histories(
            self,
            total_steps: int,
            initial_conditions: tuple[int, int, int]
    ) -> tuple[NDArray[int64], NDArray[int64], NDArray[int64], NDArray[int64], NDArray[int64]]:
        #TODO AD
        # Unpacks the initial conditions.
        initial_speed, initial_pressure, initial_temperature = initial_conditions

        # Creates empty histories.
        fuel_flow_history: NDArray[int64] = np.empty(
            total_steps + 1,
            int64
        )

        speed_history: NDArray[int64] = np.empty(
            total_steps + 1,
            int64
        )
        speed_history[0]= initial_speed

        pressure_history: NDArray[int64] = np.empty(
            total_steps + 1,
            int64
        )
        pressure_history[0]= initial_pressure

        temperature_history: NDArray[int64] = np.empty(
            total_steps + 1,
            int64
        )
        temperature_history[0]= initial_temperature

        engine_status_history: NDArray[int64] = np.empty(
            total_steps + 1,
            int64
        )

        return (
            fuel_flow_history,
            speed_history,
            pressure_history,
            temperature_history,
            engine_status_history
        )

    def run_simulation(
            self,
            initial_conditions: tuple[int, int, int]
    ) -> histories_type:
        """
        Executes the bit-true fixed-point discrete engine simulation loop.

        At each discrete time step (dt):
        1. Evaluates fuel flow schedule based on simulation time.
        2. If structural failure has tripped, cuts fuel to zero and zeros all gains.
        3. Calculates target states via fixed-point multiply-shift operations.
        4. Computes division-free discrete state derivatives using pre-calculated time factors.
        5. Advances states using forward Euler accumulation.
        6. Enforces deterministic lower clamping (N >= 0, P >= P_amb, T >= T_amb).
        7. Evaluates safety thresholds (overspeed N > N_max, thermal meltdown T > T_max).
        8. Exports complete input/output histories to CSV for hardware co-simulation assertions.

        Args:
            initial_conditions (tuple[int, int, int]): Initial state vector at t = 0 in raw LSBs:
                - initial_speed (int): Initial rotational speed (N[0]).
                - initial_pressure (int): Initial compressor pressure (P[0]).
                - initial_temperature (int): Initial exhaust gas temperature (T[0]).

        Returns:
            histories_type: Tuple of 6 NumPy arrays containing:
                - fuel_flow_history (NDArray[int64]): Quantized fuel flow over time [LSBs].
                - speed_history (NDArray[int64]): Quantized rotational speed history [LSBs].
                - pressure_history (NDArray[int64]): Quantized compressor pressure history [LSBs].
                - temperature_history (NDArray[int64]): Quantized exhaust gas temperature history [LSBs].
                - engine_status_history (NDArray[int64]): Structural failure flag history.
                - time_history (NDArray[float64]): Deterministic simulation timestamps in seconds.
        """
        # Calculates the total number of iterations needed.
        total_steps = int(
            np.ceil(self._cfg.total_simulation_time / self._cfg.simulation_step)
        )

        # Creates a time history array.
        time_history: NDArray[float64] = self._create_time_history(total_steps = total_steps)

        # Creates empty histories.
        fuel_flow_history, speed_history, pressure_history, temperature_history, engine_status_history = self._initialize_histories(
            total_steps = total_steps,
            initial_conditions = initial_conditions
        )

        # ------- Simulation Loop. -------
        is_engine_broken: bool = False

        # Defines local variables for the gains, so they can be modified if the engine breaks.
        speed_gain_coeff: int = self._cfg.speed_gain_coeff_quantized
        compressor_pressure_gain: int = self._cfg.compressor_pressure_gain_quantized
        combustion_heat_gain: int = self._cfg.combustion_heat_gain_quantized
        mass_overflow_cooling_gain: int = self._cfg.mass_overflow_cooling_gain_quantized

        for i in range(total_steps):
            # Calculates the current physical simulation time.
            k: float = i * self._cfg.simulation_step

            current_fuel_flow: int

            # Modifies the system behaviour, IF engine is broken.
            if is_engine_broken:
                # Simulates a cut-off.
                current_fuel_flow = 0

                speed_gain_coeff = 0
                compressor_pressure_gain = 0
                combustion_heat_gain = 0
                mass_overflow_cooling_gain = 0

            else:
                # Fetches the inputs.
                current_fuel_flow = self._fetch_current_fuel_flow(k = k)

            engine_status_history[i] = is_engine_broken

            fuel_flow_history[i] = current_fuel_flow

            # Fetches the current values.
            current_speed: int64 = speed_history[i]
            current_pressure: int64 = pressure_history[i]
            current_temperature: int64 = temperature_history[i]

            # Calculates the target values.
            target_speed, target_pressure, target_temperature = self._engine_datapath_fixed_point._calculate_target_values(
                current_fuel_flow = current_fuel_flow,
                current_rotational_speed = current_speed,
                gain_constants = (speed_gain_coeff, compressor_pressure_gain, combustion_heat_gain, mass_overflow_cooling_gain)
            )

            # Calculates the derivatives.
            current_values: quantized_values_type = (current_speed, current_pressure, current_temperature)

            d_speed, d_pressure, d_temperature = self._engine_datapath_fixed_point._calculate_derivatives(
                current_values = current_values,
                target_values = (target_speed, target_pressure, target_temperature)
            )

            # Calculates the next step values.
            next_speed, next_pressure, next_temperature = self._engine_datapath_fixed_point._calculate_next_step(
                current_values = current_values,
                derivative_values = (d_speed, d_pressure, d_temperature)
            )

            # ------- Controls the next values. -------
            # Clamps the results to minimum safeguards.
            next_speed = max(next_speed, 0)

            next_pressure = max(next_pressure, self._cfg.ambient_pressure_quantized)

            next_temperature = max(next_temperature, self._cfg.ambient_temperature_quantized)

            # Monitors for maximum limit breaches.
            if (current_speed > self._cfg.max_rotational_speed_quantized) or \
                (current_temperature > self._cfg.max_compressor_temperature_quantized):
                is_engine_broken = True

            speed_history[i+1] = next_speed
            pressure_history[i+1] = next_pressure
            temperature_history[i+1] = next_temperature

        # Adds the last state of fuel flow and engine state to the history.
        last_fuel_flow: int = self._fetch_current_fuel_flow(k = (total_steps * self._cfg.simulation_step))

        fuel_flow_history[total_steps] = last_fuel_flow

        engine_status_history[total_steps] = is_engine_broken

        # Packs all the histories in a single tuple.
        histories: histories_type = (fuel_flow_history, speed_history, pressure_history, temperature_history, engine_status_history, time_history)

        # Exports the input and output histories, so the data can be used as a golden reference, when compared to hardware outputs.
        self._export_histories(histories = histories)

        return histories