
#* sim_model/verify_physical_realism.py

#* ======= Libraries =======
# ------- Natives. -------
from dataclasses import dataclass
from pathlib import Path

# ------- Externals. -------
import numpy as np
import pandas as pd

from numpy import int64
from numpy.typing import NDArray
from pandas import DataFrame

# ------- Customs. -------
from sim_model.engine_datapath_fixed_point import EngineDatapathFixedPoint

#* ======= Config Physical Realism Verifier Dataclass =======
histories_type = tuple[
    NDArray[int64],
    NDArray[int64],
    NDArray[int64],
    NDArray[int64]
]
quantized_values_type = tuple[int, int, int]

@dataclass(frozen = True, slots = True)
class ConfigPhysicalRealismVerifier:
    """
    Configuration parameters governing the empirical SR-30 physical realism verification.

    Holds pre-quantized thermodynamic gains, discrete time parameters, benchmark dataset
    paths, and file export targets used to validate the digital twin against experimental
    micro-turbojet literature.

    Attributes:
        sr30_benchmark_csv_path (Path): Filesystem path to the empirical SR-30 steady-state CSV dataset.
        simulation_step (float): Discrete numerical Euler integration step size (dt) in seconds [s].
        plateau_duration (float): Hold duration per operating point to allow dynamic transients
            to decay into steady-state equilibrium ($> 4\tau$) [s].
        speed_gain_coeff_quantized (int): Pre-quantized speed gain coefficient ($K_N$) mapping fuel
            flow to rotational spool speed [raw $Q16.16$ LSBs].
        compressor_pressure_gain_quantized (int): Pre-quantized compressor pressure gain ($K_P$) mapping
            speed squared to outlet pressure [raw $Q16.16$ LSBs].
        combustion_heat_gain_quantized (int): Pre-quantized combustion thermal gain ($K_{\text{heat}}$)
            mapping fuel flow to thermal release [raw $Q16.16$ LSBs].
        mass_overflow_cooling_gain_quantized (int): Pre-quantized airflow cooling gain ($K_{\text{cool}}$)
            mapping spool rotation to mass-airflow cooling [raw $Q16.16$ LSBs].
        physical_realism_results_file_path (Path): Filesystem path where simulated equilibrium
            operating line telemetry is exported.
    """
    sr30_benchmark_file_path: Path

    simulation_step: float
    plateau_duration: float

    speed_gain_coeff_quantized: int
    compressor_pressure_gain_quantized: int
    combustion_heat_gain_quantized: int
    mass_overflow_cooling_gain_quantized: int

    physical_realism_results_file_path: Path

#* ======= Physical Realism Verifier Class =======
class PhysicalRealismVerifier:
    """
    Domain fidelity and physical realism verification harness for the jet engine digital twin.

    Executes a multi-stage throttle staircase profile derived from empirical SR-30 micro-turbojet
    literature (Bhartu 2017 / Torres et al. 2013). Integrates each discrete throttle plateau
    forward in time using the bit-true fixed-point datapath until mechanical and thermal transients
    settle into steady state, capturing the simulated equilibrium operating line for validation
    against physical engine testbed data.
    """
    def __init__(
            self,
            config_physical_realism_verifier: ConfigPhysicalRealismVerifier,
            engine_datapath_fixed_point: EngineDatapathFixedPoint
    ) -> None:
        """
        Initializes the PhysicalRealismVerifier harness with configuration and datapath components.

        Args:
            config_physical_realism_verifier (ConfigPhysicalRealismVerifier): Configuration container
                holding benchmark paths, integration steps, hold durations, and quantized gains.
            engine_datapath_fixed_point (EngineDatapathFixedPoint): Bit-true arithmetic datapath instance
                executing fixed-point target calculations, derivatives, and saturation clamping.

        Returns:
            None
        """
        self._cfg: ConfigPhysicalRealismVerifier = config_physical_realism_verifier
        self._engine_datapath_fixed_point: EngineDatapathFixedPoint = engine_datapath_fixed_point

    def _load_benchmark(self) -> tuple[NDArray[int64], quantized_values_type]:
        """
        Loads the empirical benchmark CSV at the system boundary and extracts canonical arrays.

        Parses the reference SR-30 dataset, isolates the commanded fuel flow column as a contiguous
        NumPy integer array, and extracts the baseline idle initial condition vector (speed, pressure,
        temperature) from the first operating point.

        Returns:
            tuple[NDArray[int64], quantized_values_type]: A 2-element tuple containing:
                - fuel_flow_history (NDArray[int64]): Commanded benchmark fuel flow rates [raw $Q16.16$ LSBs].
                - initial_conditions (quantized_values_type): Starting state vector (N0, P0, T0) [raw $Q16.16$ LSBs].
        """
        # Loads the csv file as a DataFrame.
        df: DataFrame = pd.read_csv(self._cfg.sr30_benchmark_file_path)

        # Converts external tabular format to arrays.
        fuel_flow_history: NDArray[int64] = df["fuel_flow"].to_numpy(dtype=int64)

        initial_conditions: quantized_values_type = (
            int(df["speed"].iloc[0]),
            int(df["pressure"].iloc[0]),
            int(df["temperature"].iloc[0]),
        )

        return fuel_flow_history, initial_conditions

    def _initialize_output_histories( #TODO Merge this method with the one from 'AircraftEngineSimulatorFİxedPoint'.
            self,
            total_points: int
    ) -> tuple[NDArray[int64], NDArray[int64], NDArray[int64]]:
        """
        Pre-allocates contiguous memory arrays for simulated equilibrium operating line capture.

        Adheres to NASA/JPL Rule 3 (no dynamic heap allocation during execution) by sizing
        output buffers prior to numerical solver loop execution.

        Args:
            total_points (int): Total number of discrete steady-state benchmark operating plateaus.

        Returns:
            tuple[NDArray[int64], NDArray[int64], NDArray[int64]]: A 3-element tuple of pre-allocated
                contiguous integer arrays for rotational speed, compressor pressure, and exhaust gas temperature.
        """
        # Creates empty histories.
        speed_history: NDArray[int64] = np.empty(
            total_points,
            int64
        )
        pressure_history: NDArray[int64] = np.empty(
            total_points,
            int64
        )
        temperature_history: NDArray[int64] = np.empty(
            total_points,
            int64
        )

        return (
            speed_history,
            pressure_history,
            temperature_history
        )

    def _export_histories( #TODO Merge this as well.
            self,
            histories: histories_type
    ) -> None:
        """
        Exports simulated steady-state plateau results to a reference CSV dataset.

        Constructs a structured pandas DataFrame containing commanded fuel flow rates and the
        corresponding settled steady-state telemetry outputs, serializing them to the configured CSV path.

        Args:
            histories (histories_type): 4-tuple of NumPy arrays containing commanded fuel flows,
                settled rotational speeds, compressor pressures, and exhaust gas temperatures.

        Returns:
            None
        """
        # Unpacks the histories.
        fuel_flow_history, speed_history, pressure_history, temperature_history = histories

        # Declares a pandas dataframe.
        df = DataFrame(
            data = {
                "fuel_flow": fuel_flow_history,
                "speed": speed_history,
                "pressure": pressure_history,
                "temperature": temperature_history
            }
        )

        # Exports the dataframe as a csv file.
        df.to_csv(
            self._cfg.physical_realism_results_file_path,
            index = False
        )

    def run_physical_realism_verifier(self) -> histories_type:
        """
        Executes the empirical physical realism verification sequence across all benchmark plateaus.

        Sequentially steps through each benchmark fuel flow command, integrating the continuous
        differential equations across the plateau hold duration using forward Euler accumulation
        to guarantee transient decay ($> 4\tau$). Captures the final settled state of each plateau
        to construct the steady-state operating line, exporting results to CSV.

        Returns:
            histories_type: 4-tuple of contiguous integer arrays containing commanded fuel flows,
                settled rotational speeds, compressor pressures, and exhaust gas temperatures
                in raw bit-true $Q16.16$ fixed-point representation.
        """
        # Loads the benchmark.
        fuel_flow_history, initial_conditions = self._load_benchmark()

        # Derives the points from the benchmark.
        total_points: int = len(fuel_flow_history)
        plateau_steps = int(np.ceil(self._cfg.plateau_duration / self._cfg.simulation_step))

        # Pre-allocates arrays.
        (   final_speed_history,
            final_pressure_history,
            final_temperature_history
        ) = self._initialize_output_histories(total_points = total_points)

        # Packs physical gains.
        gain_constants: tuple[int, int, int, int] = (
            self._cfg.speed_gain_coeff_quantized,
            self._cfg.compressor_pressure_gain_quantized,
            self._cfg.combustion_heat_gain_quantized,
            self._cfg.mass_overflow_cooling_gain_quantized
        )

        # Initializes current parameters with starting idle conditions.
        current_speed, current_pressure, current_temperature = initial_conditions

        # ------- Simulation Loop. -------
        for data_collecting_point in range(total_points):
            current_fuel_flow = int(fuel_flow_history[data_collecting_point])

            for _ in range(plateau_steps):
                # Calculates the target values.
                target_values: quantized_values_type = self._engine_datapath_fixed_point.calculate_target_values(
                    current_fuel_flow = current_fuel_flow,
                    current_rotational_speed = current_speed,
                    gain_constants = gain_constants
                )

                # Calculates the derivatives.
                current_values: quantized_values_type = (current_speed, current_pressure, current_temperature)

                d_values: quantized_values_type = self._engine_datapath_fixed_point.calculate_derivatives(
                    current_values = current_values,
                    target_values = target_values
                )

                # Calculates the next step values.
                (   next_speed,
                    next_pressure,
                    next_temperature
                ) = self._engine_datapath_fixed_point.calculate_next_step(
                    current_values = current_values,
                    derivative_values = d_values
                )

                # Updates the current parameters.
                current_speed = next_speed
                current_pressure = next_pressure
                current_temperature = next_temperature

            # Adds the last state current parameters to final histories.
            final_speed_history[data_collecting_point] = current_speed
            final_pressure_history[data_collecting_point] = current_pressure
            final_temperature_history[data_collecting_point] = current_temperature

        # Packs all the histories in a single tuple.
        histories: histories_type = (
            fuel_flow_history,
            final_speed_history,
            final_pressure_history,
            final_temperature_history
        )

        # Exports the input and output histories.
        self._export_histories(histories = histories)

        return histories