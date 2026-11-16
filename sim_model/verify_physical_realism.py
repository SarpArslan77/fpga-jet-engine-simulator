
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
    sr30_benchmark_csv_path: Path

    simulation_step: float

    plateau_duration: float

    speed_gain_coeff_quantized: int
    compressor_pressure_gain_quantized: int
    combustion_heat_gain_quantized: int
    mass_overflow_cooling_gain_quantized: int

#* ======= Physical Realism Verifier Class =======
class PhysicalRealismVerifier:
    def __init__(
            self,
            config_physical_realism_verifier: ConfigPhysicalRealismVerifier,
            engine_datapath_fixed_point: EngineDatapathFixedPoint
    ) -> None:
        self._cfg: ConfigPhysicalRealismVerifier = config_physical_realism_verifier
        self._engine_datapath_fixed_point: EngineDatapathFixedPoint = engine_datapath_fixed_point

    def _load_benchmark(self) -> DataFrame:
        # Loads the csv file as a DataFrame.
        df: DataFrame = pd.read_csv(self._cfg.sr30_benchmark_csv_path)

        return df

    def _initialize_histories( #TODO Merge this method with the one from 'AircraftEngineSimulatorFİxedPoint'.
            self,
            total_steps: int,
            initial_conditions: tuple[int, int, int],
            fuel_flow_col_df: DataFrame,
    ) -> histories_type:
        # Fetches the initial conditions.
        initial_speed, initial_pressure, initial_temperature = initial_conditions

        # Fetches the fuel flow history from the csv file.
        fuel_flow_history: NDArray[int64] = fuel_flow_col_df.to_numpy(dtype = int64)

        # Creates empty histories.
        speed_history: NDArray[int64] = np.empty(
            total_steps + 1,
            int64
        )
        speed_history[0] = initial_speed

        pressure_history: NDArray[int64] = np.empty(
            total_steps + 1,
            int64
        )
        pressure_history[0] = initial_pressure

        temperature_history: NDArray[int64] = np.empty(
            total_steps + 1,
            int64
        )
        temperature_history[0] = initial_temperature

        return (
            fuel_flow_history,
            speed_history,
            pressure_history,
            temperature_history
        )

    def _export_histories( #TODO Merge this as well.
            self,
            histories: histories_type
    ) -> None:
        # Unpacks the histories.
        _, speed_history, pressure_history, temperature_history = histories

        # Declares a pandas dataframe.
        df = DataFrame(
            data = {
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

    def run_physical_realism_verifier(self) -> None:
        # Defines the loop boundaries.
        data_collecting_point_count: int = 15 #TODO FTH
        plateau_total_step_count: int = int(self._cfg.plateau_duration / self._cfg.simulation_step)

        # Loads the csv file as a DataFrame.
        sr30_benchmark_df: DataFrame = self._load_benchmark()

        # Fetches the fuel flow history and initial conditions.
        fuel_flow_col_df: DataFrame = sr30_benchmark_df["fuel_flow"]

        initial_speed: int64 = sr30_benchmark_df["speed"].iloc[0]
        initial_pressure: int64 = sr30_benchmark_df["pressure"].iloc[0]
        initial_temperature: int64 = sr30_benchmark_df["temperature"].iloc[0]

        # Creates final-empty histories with initial conditions.
        fuel_flow_history, final_speed_history, final_pressure_history, final_temperature_history = self._initialize_histories(
            total_steps = data_collecting_point_count,
            initial_conditions = (initial_speed, initial_pressure, initial_temperature),
            fuel_flow_col_df = fuel_flow_col_df
        )

        # ------- Simulation Loop. -------
        for data_collecting_point in range(data_collecting_point_count-1):
            # Fetches the current values.
            current_fuel_flow = int(fuel_flow_history[data_collecting_point])
            current_speed = int(final_speed_history[data_collecting_point])
            current_pressure = int(final_pressure_history[data_collecting_point])
            current_temperature = int(final_temperature_history[data_collecting_point])

            # Creates intermediate-empty histories with initial conditions.
            _, intermediate_speed_history, intermediate_pressure_history, intermediate_temperature_history = self._initialize_histories(
                total_steps = plateau_total_step_count,
                initial_conditions = (final_speed_history[data_collecting_point], final_pressure_history[data_collecting_point], final_temperature_history[data_collecting_point]),
                fuel_flow_col_df = fuel_flow_col_df #TODO Find a method where this does not get passed, since its redundant.
            )

            for i in range(plateau_total_step_count):
                # Calculates the target values.
                target_speed, target_pressure, target_temperature = self._engine_datapath_fixed_point._calculate_target_values(
                    current_fuel_flow = current_fuel_flow,
                    current_rotational_speed = current_speed,
                    gain_constants = (self._cfg.speed_gain_coeff_quantized, self._cfg.compressor_pressure_gain_quantized, self._cfg.combustion_heat_gain_quantized, self._cfg.mass_overflow_cooling_gain_quantized)
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

                # Stores the intermediate-results.
                intermediate_speed_history[i+1] = next_speed
                intermediate_pressure_history[i+1] = next_pressure
                intermediate_temperature_history[i+1] = next_temperature

            # Adds the last state of intermediate histories to final histories.
            final_speed_history[data_collecting_point+1] = intermediate_speed_history[-1]
            final_pressure_history[data_collecting_point+1] = intermediate_pressure_history[-1]
            final_temperature_history[data_collecting_point+1] = intermediate_temperature_history[-1]