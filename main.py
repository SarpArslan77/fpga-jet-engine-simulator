#* main.py

#* ======= Libraries =======
# ------- Natives. -------
import json
import numpy as np
import os
import shutil
import subprocess
import sys

from argparse import ArgumentParser, Namespace
from pathlib import Path
from subprocess import CompletedProcess
from typing import Any

# ------- Externals. -------
import pandas as pd

#? get_runner(): Factory function, returns a simulator-specific controller.
from cocotb_tools.runner import Runner, get_runner
from numpy import float64
from numpy.typing import NDArray
from pandas import DataFrame

# ------- Customs. -------
from scripts.config_loader import ConfigLoader, SystemConfig
from scripts.generate_verilog_header_file import generate_verilog_header_file
from scripts.locate_executables import ConfigExecutableLocater, ExecutableLocater
from scripts.run_synthesis import ConfigSynthesisRunner, SynthesisRunner
from sim_model.aircraft_engine_simulator_fixed_point import (
    AircraftEngineSimulatorFixedPoint,
    ConfigAircraftEngineSimulatorFixedPoint,
)
from sim_model.aircraft_engine_simulator_floating import (
    AircraftEngineSimulatorFloating,
    ConfigAircraftEngineSimulatorFloating,
)
from sim_model.engine_datapath_fixed_point import ConfigEngineDatapathFixedPoint, EngineDatapathFixedPoint
from sim_model.graph_visualizer import ConfigGraphVisualizer, GraphVisualizer
from sim_model.verify_physical_realism import ConfigPhysicalRealismVerifier, PhysicalRealismVerifier
from sim_model.verify_software import SoftwareVerifier

#* ======= Main Pipeline =======
if __name__ == "__main__":
    #* -------------------------------------------------------------------------
    #* [STAGE 1] System Paths & Root Configuration (CSST)
    #* -------------------------------------------------------------------------
    # System Paths.
    MAIN_DIR_PATH: Path = Path(__file__).resolve().parent
    RTL_DIR_PATH: Path = MAIN_DIR_PATH / "rtl"
    TB_DIR_PATH: Path = MAIN_DIR_PATH / "tb"
    BUILD_SYNTH_DIR_PATH: Path = MAIN_DIR_PATH / "build_synth"
    SCRIPTS_DIR_PATH: Path = MAIN_DIR_PATH / "scripts"
    DATA_DIR_PATH: Path = MAIN_DIR_PATH / "data"

    CONFIG_FILE_PATH: Path = MAIN_DIR_PATH / "config.toml"
    TCL_SCRIPT_PATH: Path = SCRIPTS_DIR_PATH / "synth.tcl"
    README_FILE_PATH: Path = MAIN_DIR_PATH / "README.md"
    SOFTWARE_RESULTS_FILE_PATH: Path = MAIN_DIR_PATH / "software_results.csv"
    SR30_BENCHMARK_FILE_PATH: Path = DATA_DIR_PATH / "sr30_benchmark.csv"

    # Hardware Source Definitions.
    TOP_MODULE: str = "aircraft_engine_simulator_top"
    VERILOG_SRC_FILE_PATHS: list[Path] = [
        RTL_DIR_PATH / "clock_prescaler.v",
        RTL_DIR_PATH / "calculate_target_values.v",
        RTL_DIR_PATH / "calculate_derivatives.v",
        RTL_DIR_PATH / "calculate_next_steps.v",
        RTL_DIR_PATH / "axi4_lite.v",
        RTL_DIR_PATH / f"{TOP_MODULE}.v",
    ]

    # Central Single Source of Truth (CSST).
    config_loader = ConfigLoader()
    cfg: SystemConfig = config_loader.load_config(config_file_path=CONFIG_FILE_PATH)

    # Derived Physics Parameters.
    slowest_time_const: float = max(
        cfg.inertia.rotor_inertial_time_const,
        cfg.inertia.pressure_volume_time_const,
        cfg.inertia.thermal_time_const,
    )
    # Dynamic plateau hold duration ensuring transient decay (> 4*tau).
    settling_duration: float = slowest_time_const * (-np.log(cfg.inertia.settling_band))
    plateau_duration: float = float(np.ceil(settling_duration))

    #* -------------------------------------------------------------------------
    #* [STAGE 2] Command-Line Interface & Argument Parsing
    #* -------------------------------------------------------------------------
    parser = ArgumentParser(description="Aircraft Engine Simulation & Verification Pipeline.")

    # Mutually exclusive synthesis triggers: either a single target run OR a frequency sweep.
    synthesizer_group = parser.add_mutually_exclusive_group()

    synthesizer_group.add_argument(
        "--synth",
        nargs = "?",
        const = cfg.synthesis.synth_target_period,
        type = float,
        default = None,
        help = f"Run single Out-of-Context synthesis. Optional target period in [ns] (default: {cfg.synthesis.synth_target_period}).",
    )

    synthesizer_group.add_argument(
        "--sweep",
        nargs = "*",
        type = float,
        default = None,
        metavar = ("LOW", "HIGH"),
        help = "Execute binary search frequency sweep. Pass 0 arguments for defaults or 2 bounds: LOW HIGH in [ns].",
    )

    parser.add_argument(
        "--lint",
        action = "store_true",
        help = "Run Ruff, Verible, Verilator, and Mypy static quality checks.",
    )

    args: Namespace = parser.parse_args()

    #* -------------------------------------------------------------------------
    #* [STAGE 3] Toolchain Locating & Environment Provisioning
    #* -------------------------------------------------------------------------
    config_executable_locater = ConfigExecutableLocater(
        hardcoded_iverilog_path = Path("C:/iverilog/bin"),
        hardcoded_verilator_path = Path("D:/Programs/oss-cad-suite"),
        hardcoded_vivado_path = Path("D:/Programs/2025.2/Vivado/bin/vivado.bat"),
    )
    executable_locater = ExecutableLocater(config_executable_locater=config_executable_locater)

    # Locates Icarus Verilog and adds to system PATH if found locally.
    iverilog_executable_path: Path | None = None
    try:
        iverilog_executable_path = executable_locater.locate_iverilog_executable()
        iverilog_bin_dir: str = str(iverilog_executable_path.parent)

        if iverilog_bin_dir not in os.environ.get("PATH", ""):
            os.environ["PATH"] = iverilog_bin_dir + os.pathsep + os.environ.get("PATH", "")
    except FileNotFoundError:
        print("[NOTE] Icarus Verilog not found locally (hardware sim will be enforced in CI).")

    # Silences Python warning filter for Cocotb runner.
    os.environ["PYTHONWARNINGS"] = "ignore::DeprecationWarning"

    #* -------------------------------------------------------------------------
    #* [STAGE 4] Hardware Code Generation & Inter-Module Export
    #* -------------------------------------------------------------------------
    # Exports Testbench Configuration (tb/config.json).
    tb_config: dict[str, dict[str, Any]] = {
        "test_engine": {
            "addr_control": cfg.register.addr_control,
            "addr_status": cfg.register.addr_status,
            "addr_fuel_flow_cmd": cfg.register.addr_fuel_flow_cmd,
            "addr_speed": cfg.register.addr_speed,
            "addr_pressure": cfg.register.addr_pressure,
            "addr_temperature": cfg.register.addr_temperature,
            "addr_fault_injection": cfg.register.addr_fault_injection,
            "software_results_file_path_str": str(SOFTWARE_RESULTS_FILE_PATH),
            "simulation_step": cfg.simulation.simulation_step,
            "total_simulation_time": cfg.simulation.total_simulation_time,
            "fractional_bit_width": cfg.representation.fractional_bit_width,
            "max_compressor_temperature": cfg.limit.max_compressor_temperature,
            "smoke_test_duration": cfg.testbench.smoke_test_duration,
            "compressor_stall_test_duration": cfg.testbench.compressor_stall_test_duration,
            "max_thermal_threshold_cross_duration": cfg.testbench.max_thermal_threshold_cross_duration,
        }
    }

    target_config_path: Path = TB_DIR_PATH / "config.json"
    try:
        target_config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file=target_config_path, mode="w", encoding="utf-8") as f:
            json.dump(tb_config, f, indent=4)
    except TypeError as err:
        raise TypeError(f"Failed to serialize testbench configurations into JSON format: {err}") from err
    except OSError as err:
        raise RuntimeError(f"Failed to write testbench configuration to '{target_config_path}': {err}") from err

    # Synthesizable Parameter Dictionaries.
    ENGINE_MATH_PARAMETERS = {
        "INT_BIT_WIDTH": cfg.representation.int_bit_width,
        "FRACTIONAL_BIT_WIDTH": cfg.representation.fractional_bit_width,
        "AMBIENT_PRESSURE": cfg.ambient.ambient_pressure_quantized,
        "AMBIENT_TEMPERATURE": cfg.ambient.ambient_temperature_quantized,
        "SPEED_GAIN_COEFF": cfg.gain.speed_gain_coeff_quantized,
        "COMPRESSOR_PRESSURE_GAIN": cfg.gain.compressor_pressure_gain_quantized,
        "COMBUSTION_HEAT_GAIN": cfg.gain.combustion_heat_gain_quantized,
        "MASS_OVERFLOW_COOLING_GAIN": cfg.gain.mass_overflow_cooling_gain_quantized,
        "PRE_CALCULATED_ROTOR_INERTIAL_TIME_CONST": cfg.inertia.pre_calculated_rotor_inertial_time_const_quantized,
        "PRE_CALCULATED_PRESSURE_VOLUME_TIME_CONST": cfg.inertia.pre_calculated_pressure_volume_time_const_quantized,
        "PRE_CALCULATED_THERMAL_TIME_CONST": cfg.inertia.pre_calculated_thermal_time_const_quantized,
        "MAX_ROTATIONAL_SPEED": cfg.limit.max_rotational_speed_quantized,
        "MAX_COMPRESSOR_TEMPERATURE": cfg.limit.max_compressor_temperature_quantized,
    }

    COCOTB_PARAMETERS = {
        **ENGINE_MATH_PARAMETERS,
        "SYSTEM_CLK_FREQ": 25,
        "TARGET_CLK_FREQ": 1,
    }

    TOP_MODULE_PARAMETERS = {
        **ENGINE_MATH_PARAMETERS,
        "SYSTEM_CLK_FREQ": 100_000_000,
        "TARGET_CLK_FREQ": 1_000,
    }

    # Generates Verilog Parameter Include Header (rtl/top_module_params.vh).
    generate_verilog_header_file(
        file_name = "top_module_params.vh",
        parameters = TOP_MODULE_PARAMETERS,
        verilog_header_file_dir_path = RTL_DIR_PATH,
    )

    #* -------------------------------------------------------------------------
    #* [STAGE 5] Static Analysis & Code Quality Gates (Optional Flag)
    #* -------------------------------------------------------------------------
    if args.lint:
        print()
        print("Running linters (Ruff, Verible, Verilator, and Mypy)...")

        # -> Ruff Python Lint.
        ruff_status: str
        ruff_path: str | None = shutil.which(cmd="ruff")
        if ruff_path is not None:
            ruff_cmd: list[str] = ["ruff", "check", "."]
            ruff_result: CompletedProcess[str] = subprocess.run(ruff_cmd, cwd=MAIN_DIR_PATH, text=True)
            ruff_status = "SUCCESS" if ruff_result.returncode == 0 else "FAILED"
            print(f"-> [{ruff_status}] Ruff")
        else:
            ruff_status = "NOT FOUND"
            print(f"[{ruff_status}] Ruff not found locally (enforced via GitHub Action 'Ruff').")

        # -> Verible Verilog Syntax Lint.
        verible_status: str
        verible_path: str | None = shutil.which(cmd="verible-verilog-lint")
        if verible_path is not None:
            verilog_file_paths: list[str] = [str(file_path) for file_path in VERILOG_SRC_FILE_PATHS]
            verible_cmd: list[str] = [
                "verible-verilog-lint",
                "--rules_config",
                ".rules.verible_lint",
            ] + verilog_file_paths
            verible_result: CompletedProcess[str] = subprocess.run(verible_cmd, cwd=MAIN_DIR_PATH, text=True)
            verible_status = "SUCCESS" if verible_result.returncode == 0 else "FAILED"
            print(f"-> [{verible_status}] Verible")
        else:
            verible_status = "NOT FOUND"
            print(f"[{verible_status}] Verible not found locally (enforced via GitHub Action 'Verible').")

        # -> Verilator DO-254 Semantic Lint.
        verilator_status: str
        verilator_executable_path: Path | None = None
        try:
            verilator_executable_path = executable_locater.locate_verilator_executable()
            verilator_bin_dir: str = str(verilator_executable_path.parent)

            if verilator_bin_dir not in os.environ.get("PATH", ""):
                os.environ["PATH"] = verilator_bin_dir + os.pathsep + os.environ.get("PATH", "")
        except FileNotFoundError:
            print()
            print("[NOTE] Verilator not found locally (DO-254 linting will be enforced in CI).")

        if verilator_executable_path is not None:
            verilator_cmd: list[str] = [
                str(verilator_executable_path),
                "--lint-only",
                "-Wall",
                "-Wwarn-style",
                "-Werror-WIDTH",
                "-Werror-UNOPTFLAT",
                "-Werror-CASEINCOMPLETE",
                "-Werror-LATCH",
                "-I" + str(RTL_DIR_PATH),
                str(RTL_DIR_PATH / f"{TOP_MODULE}.v"),
            ]
            verilator_result: CompletedProcess[str] = subprocess.run(verilator_cmd, cwd=MAIN_DIR_PATH, text=True)
            verilator_status = "SUCCESS" if verilator_result.returncode == 0 else "FAILED"
            print(f"-> [{verilator_status}] Verilator")
        else:
            verilator_status = "NOT FOUND"
            print(f"[{verilator_status}] Verilator not found locally (enforced via GitHub Action 'Verilator').")

        # -> Mypy Strict Static Type Check.
        mypy_status: str
        mypy_path: str | None = shutil.which(cmd="mypy")
        if mypy_path is not None:
            mypy_cmd: list[str] = [
                "mypy",
                "--config-file",
                "pyproject.toml",
                "scripts",
                "sim_model",
                "tb",
                "main.py",
            ]
            mypy_result: CompletedProcess[str] = subprocess.run(mypy_cmd, cwd=MAIN_DIR_PATH, text=True)
            mypy_status = "SUCCESS" if mypy_result.returncode == 0 else "FAILED"
            print(f"-> [{mypy_status}] Mypy")
        else:
            mypy_status = "NOT FOUND"
            print(f"[{mypy_status}] Mypy not found locally (enforced via GitHub Action 'Mypy').")

        # Enforces zero-warning policy across linters.
        if (ruff_status != "FAILED") and (verible_status != "FAILED") and (verilator_status != "FAILED"):
            print("--> Linting passed! <--")
        else:
            print("--> Linting failed! <--")
            sys.exit(1)

    #* -------------------------------------------------------------------------
    #* [STAGE 6] Subsystem Instantiations
    #* -------------------------------------------------------------------------
    # -> Floating-Point Continuous Model.
    config_aircraft_engine_simulator_floating = ConfigAircraftEngineSimulatorFloating(
        simulation_step = cfg.simulation.simulation_step,
        total_simulation_time = cfg.simulation.total_simulation_time,
        ambient_pressure = cfg.ambient.ambient_pressure,
        ambient_temperature = cfg.ambient.ambient_temperature,
        rotor_inertial_time_const = cfg.inertia.rotor_inertial_time_const,
        pressure_volume_time_const = cfg.inertia.pressure_volume_time_const,
        thermal_time_const = cfg.inertia.thermal_time_const,
        speed_gain_coeff = cfg.gain.speed_gain_coeff,
        compressor_pressure_gain = cfg.gain.compressor_pressure_gain,
        combustion_heat_gain = cfg.gain.combustion_heat_gain,
        mass_overflow_cooling_gain = cfg.gain.mass_overflow_cooling_gain,
        max_rotational_speed = cfg.limit.max_rotational_speed,
        max_compressor_temperature = cfg.limit.max_compressor_temperature,
    )
    aircraft_engine_simulator_floating = AircraftEngineSimulatorFloating(
        config_aircraft_engine_simulator_floating = config_aircraft_engine_simulator_floating
    )

    # -> Engine Datapath Fixed Point.
    config_engine_datapath_fixed_point = ConfigEngineDatapathFixedPoint(
        fractional_bit_width = cfg.representation.fractional_bit_width,
        ambient_pressure_quantized = cfg.ambient.ambient_pressure_quantized,
        ambient_temperature_quantized = cfg.ambient.ambient_temperature_quantized,
        pre_calculated_rotor_inertial_time_const_quantized = cfg.inertia.pre_calculated_rotor_inertial_time_const_quantized,
        pre_calculated_pressure_volume_time_const_quantized = cfg.inertia.pre_calculated_pressure_volume_time_const_quantized,
        pre_calculated_thermal_time_const_quantized = cfg.inertia.pre_calculated_thermal_time_const_quantized
    )
    engine_datapath_fixed_point = EngineDatapathFixedPoint(
        config_engine_datapath_fixed_point = config_engine_datapath_fixed_point
    )

    # -> Bit-True Q16.16 Fixed-Point Model.
    config_aircraft_engine_simulator_fixed_point = ConfigAircraftEngineSimulatorFixedPoint(
        int_bit_width = cfg.representation.int_bit_width,
        fractional_bit_width = cfg.representation.fractional_bit_width,
        simulation_step = cfg.simulation.simulation_step,
        total_simulation_time = cfg.simulation.total_simulation_time,
        ambient_pressure_quantized = cfg.ambient.ambient_pressure_quantized,
        ambient_temperature_quantized = cfg.ambient.ambient_temperature_quantized,
        speed_gain_coeff_quantized = cfg.gain.speed_gain_coeff_quantized,
        compressor_pressure_gain_quantized = cfg.gain.compressor_pressure_gain_quantized,
        combustion_heat_gain_quantized = cfg.gain.combustion_heat_gain_quantized,
        mass_overflow_cooling_gain_quantized = cfg.gain.mass_overflow_cooling_gain_quantized,
        max_rotational_speed_quantized = cfg.limit.max_rotational_speed_quantized,
        max_compressor_temperature_quantized = cfg.limit.max_compressor_temperature_quantized,
        software_results_file_path=SOFTWARE_RESULTS_FILE_PATH,
    )
    aircraft_engine_simulator_fixed_point = AircraftEngineSimulatorFixedPoint(
        config_aircraft_engine_simulator_fixed_point = config_aircraft_engine_simulator_fixed_point,
        engine_datapath_fixed_point = engine_datapath_fixed_point
    )

    # -> Software Verifier (Numerical Accuracy).
    software_verifier = SoftwareVerifier()

    # -> Telemetry Graph Visualizer.
    config_graph_visualizer = ConfigGraphVisualizer(
        fuel_flow_floating_linestyle = "--",
        fuel_flow_floating_color = "gray",
        fuel_flow_floating_linewidth = 1.5,
        fuel_flow_fixed_point_linestyle = ":",
        fuel_flow_fixed_point_color = "black",
        fuel_flow_fixed_point_linewidth = 1.5,
        speed_floating_linestyle = "-",
        speed_floating_color = "blue",
        speed_floating_linewidth = 2.0,
        speed_fixed_point_linestyle = "-.",
        speed_fixed_point_color = "darkblue",
        speed_fixed_point_linewidth = 2.0,
        fuel_flow_and_speed_grid_linestyle = ":",
        fuel_flow_and_speed_grid_alpha = 0.6,
        pressure_floating_linestyle = "-",
        pressure_floating_color = "purple",
        pressure_floating_linewidth = 2.0,
        pressure_fixed_point_linestyle = "-.",
        pressure_fixed_point_color = "indigo",
        pressure_fixed_point_linewidth = 2.0,
        pressure_grid_linestyle = ":",
        pressure_grid_alpha = 0.6,
        temperature_floating_linestyle = "-",
        temperature_floating_color = "red",
        temperature_floating_linewidth = 2.0,
        temperature_fixed_point_linestyle = "-.",
        temperature_fixed_point_color = "darkred",
        temperature_fixed_point_linewidth = 2.0,
        temperature_grid_linestyle = ":",
        temperature_grid_alpha = 0.6,
        max_speed = cfg.limit.max_rotational_speed,
        engine_failure_line_color = "darkred",
        engine_failure_line_linestyle = ":",
        engine_failure_line_linewidth = 1.5,
        engine_failure_text_color = "darkred",
        engine_failure_text_fontsize = 9,
        engine_failure_text_fontweight = "bold",
    )
    graph_visualizer = GraphVisualizer(config_graph_visualizer = config_graph_visualizer)

    # -> Vivado Synthesis Automation Runner.
    config_synthesis_runner = ConfigSynthesisRunner(
        main_dir_path = MAIN_DIR_PATH,
        build_synth_dir_path = BUILD_SYNTH_DIR_PATH,
        scripts_dir_path = SCRIPTS_DIR_PATH,
        tcl_script_path = TCL_SCRIPT_PATH,
        readme_file_path = README_FILE_PATH,
    )
    synthesis_runner = SynthesisRunner(config_synthesis_runner = config_synthesis_runner)

    # -> Physical Realism Verifier (Empirical SR-30 Benchmark).
    config_physical_realism_verifier = ConfigPhysicalRealismVerifier(
        sr30_benchmark_file_path = SR30_BENCHMARK_FILE_PATH,
        simulation_step = cfg.simulation.simulation_step,
        plateau_duration=plateau_duration,
        speed_gain_coeff_quantized = cfg.gain.speed_gain_coeff_quantized,
        compressor_pressure_gain_quantized = cfg.gain.compressor_pressure_gain_quantized,
        combustion_heat_gain_quantized = cfg.gain.combustion_heat_gain_quantized,
        mass_overflow_cooling_gain_quantized = cfg.gain.mass_overflow_cooling_gain_quantized,
        physical_realism_results_file_path = DATA_DIR_PATH / "physical_realism_test_data.csv",
    )
    physical_realism_verifier = PhysicalRealismVerifier(
        config_physical_realism_verifier = config_physical_realism_verifier,
        engine_datapath_fixed_point = engine_datapath_fixed_point,
    )

    #* -------------------------------------------------------------------------
    #* [STAGE 7] Numerical Reference Execution & Quantization Analysis
    #* -------------------------------------------------------------------------
    # Executes Continuous 64-Bit Float Reference Simulation.
    (   fuel_flow_history_floating,
        speed_history_floating,
        pressure_history_floating,
        temperature_history_floating,
        time_history,
    ) = aircraft_engine_simulator_floating.run_simulation(
        initial_conditions = (
            0.0,
            cfg.ambient.ambient_pressure,
            cfg.ambient.ambient_temperature
        )
    )

    # Executes Bit-True Fixed-Point Simulation.
    (   fuel_flow_history_fixed_point,
        speed_history_fixed_point,
        pressure_history_fixed_point,
        temperature_history_fixed_point,
        engine_status_history,
        _,
    ) = aircraft_engine_simulator_fixed_point.run_simulation(
        initial_conditions = (
            0,
            cfg.ambient.ambient_pressure_quantized,
            cfg.ambient.ambient_temperature_quantized,
        )
    )

    # Scales fixed-point integer LSBs back to physical engineering units for comparison.
    FIXED_POINT_SCALE_FACTOR: int = 2 ** cfg.representation.fractional_bit_width

    fuel_flow_history_fixed_point_scaled: NDArray[float64] = fuel_flow_history_fixed_point / FIXED_POINT_SCALE_FACTOR
    speed_history_fixed_point_scaled: NDArray[float64] = speed_history_fixed_point / FIXED_POINT_SCALE_FACTOR
    pressure_history_fixed_point_scaled: NDArray[float64] = pressure_history_fixed_point / FIXED_POINT_SCALE_FACTOR
    temperature_history_fixed_point_scaled: NDArray[float64] = temperature_history_fixed_point / FIXED_POINT_SCALE_FACTOR

    # Computes numerical quantization error residuals.
    print()
    print("Quantization Verification Results: ")

    software_fuel_flow_error: NDArray[float64] = fuel_flow_history_floating - fuel_flow_history_fixed_point_scaled
    software_speed_error: NDArray[float64] = speed_history_floating - speed_history_fixed_point_scaled
    software_pressure_error: NDArray[float64] = pressure_history_floating - pressure_history_fixed_point_scaled
    software_temperature_error: NDArray[float64] = temperature_history_floating - temperature_history_fixed_point_scaled

    # Statistical Evaluation (MAE, ME, MSE, RMSE, SQNR).
    software_verifier.run_software_verifier(
        original_signal = fuel_flow_history_floating,
        error = software_fuel_flow_error,
        state_name = "Fuel Flow",
    )
    software_verifier.run_software_verifier(
        original_signal = speed_history_floating,
        error = software_speed_error,
        state_name = "Rotational Speed",
    )
    software_verifier.run_software_verifier(
        original_signal = pressure_history_floating,
        error = software_pressure_error,
        state_name = "Compressor Pressure",
    )
    software_verifier.run_software_verifier(
        original_signal = temperature_history_floating,
        error = software_temperature_error,
        state_name = "Exhaust Gas Temperature",
    )

    # Renders comparative verification waveforms.
    graph_visualizer.run_visualization(
        histories_floating = (
            fuel_flow_history_floating,
            speed_history_floating,
            pressure_history_floating,
            temperature_history_floating,
        ),
        histories_fixed_point_scaled = (
            fuel_flow_history_fixed_point_scaled,
            speed_history_fixed_point_scaled,
            pressure_history_fixed_point_scaled,
            temperature_history_fixed_point_scaled,
        ),
        time_history=time_history,
        show_plots=False,
    )

    #* -------------------------------------------------------------------------
    #* [STAGE 8] Domain Fidelity & Physical Realism Verification
    #* -------------------------------------------------------------------------
    # Executes the verifier.
    (   physical_realism_fuel_flow_history,
        physical_realism_speed_history,
        physical_realism_pressure_history,
        physical_realism_temperature_history
    ) = physical_realism_verifier.run_physical_realism_verifier()

    # Scales physical-realism integer LSBs back to physical engineering units for comparison.
    physical_realism_fuel_flow_history_scaled: NDArray[float64] = physical_realism_fuel_flow_history / FIXED_POINT_SCALE_FACTOR
    physical_realism_speed_history_scaled: NDArray[float64] = physical_realism_speed_history / FIXED_POINT_SCALE_FACTOR
    physical_realism_pressure_history_scaled: NDArray[float64] = physical_realism_pressure_history / FIXED_POINT_SCALE_FACTOR
    physical_realism_temperature_history_scaled: NDArray[float64] = physical_realism_temperature_history / FIXED_POINT_SCALE_FACTOR

    # Loads the benchmark reference.
    benchmark_df: DataFrame = pd.read_csv(SR30_BENCHMARK_FILE_PATH)

    #TODO Add a function to scale the numbers up and down (quantize / unquantize). (update quantize_floats to be 2-way onsie)
    benchmark_fuel_flow_scaled: NDArray[float64] = (benchmark_df["fuel_flow"].to_numpy()) / FIXED_POINT_SCALE_FACTOR
    benchmark_speed_scaled: NDArray[float64] = (benchmark_df["speed"].to_numpy()) / FIXED_POINT_SCALE_FACTOR
    benchmark_pressure_scaled: NDArray[float64] = (benchmark_df["pressure"].to_numpy()) / FIXED_POINT_SCALE_FACTOR
    benchmark_temperature_scaled: NDArray[float64] = (benchmark_df["temperature"].to_numpy()) / FIXED_POINT_SCALE_FACTOR

    # Computes numerical physical-realism error residuals.
    print()
    print("Physical Realism Verification Results (SR-30 Benchmark):")

    physical_fuel_flow_error: NDArray[float64] = benchmark_fuel_flow_scaled - physical_realism_fuel_flow_history_scaled
    physical_speed_error: NDArray[float64] = benchmark_speed_scaled - physical_realism_speed_history_scaled
    physical_pressure_error: NDArray[float64] = benchmark_pressure_scaled - physical_realism_pressure_history_scaled
    physical_temperature_error: NDArray[float64] = benchmark_temperature_scaled - physical_realism_temperature_history_scaled

    # Statistical Evaluation (MAE, ME, MSE, RMSE, SQNR).
    software_verifier.run_software_verifier( #TODO Change name to be sth more conventional.
        original_signal = benchmark_fuel_flow_scaled,
        error = physical_fuel_flow_error,
        state_name = "Fuel Flow",
    )
    software_verifier.run_software_verifier(
        original_signal = benchmark_speed_scaled,
        error = physical_speed_error,
        state_name = "Speed",
    )
    software_verifier.run_software_verifier(
        original_signal = benchmark_pressure_scaled,
        error = physical_pressure_error,
        state_name = "Pressure",
    )
    software_verifier.run_software_verifier(
        original_signal = benchmark_temperature_scaled,
        error = physical_temperature_error,
        state_name = "Temperature",
    )

    #* -------------------------------------------------------------------------
    #* [STAGE 9] Hardware Co-Simulation (Cocotb & Icarus Verilog)
    #* -------------------------------------------------------------------------
    runner: Runner = get_runner("icarus")

    runner.build(
        sources=VERILOG_SRC_FILE_PATHS,
        hdl_toplevel=TOP_MODULE,
        includes=[RTL_DIR_PATH],
        parameters=COCOTB_PARAMETERS,
        always=True,
    )

    runner.test(
        hdl_toplevel=TOP_MODULE,
        test_module="test_engine",
        test_dir=TB_DIR_PATH,
    )

    #* -------------------------------------------------------------------------
    #* [STAGE 10] Automated Vivado Synthesis & Timing Closure (OOC)
    #* -------------------------------------------------------------------------
    vivado_path: Path

    # Binary search clock frequency sweep.
    if args.sweep is not None:
        if len(args.sweep) == 2:
            adjusted_low_period_ns: float = float(min(args.sweep))
            adjusted_high_period_ns: float = float(max(args.sweep))
        elif len(args.sweep) == 0:
            adjusted_low_period_ns = cfg.synthesis.sweep_low_period
            adjusted_high_period_ns = cfg.synthesis.sweep_high_period
        else:
            parser.error(
                message="--sweep accepts either 0 arguments (using defaults) or exactly 2 bounds: LOW HIGH in [ns]."
            )

        print()
        print(f"Calculating maximum frequency across period bounds [{adjusted_low_period_ns}, {adjusted_high_period_ns}] ns...")

        best_synthesizable_period: float = adjusted_high_period_ns
        best_wns: float = -np.inf

        while (adjusted_high_period_ns - adjusted_low_period_ns) > 0.100:
            test_period_ns: float = round(((adjusted_high_period_ns + adjusted_low_period_ns) / 2), 3)
            vivado_path = executable_locater.locate_vivado_executable()

            wns: float = synthesis_runner.run_synthesis(
                target_period_ns=test_period_ns,
                should_update_readme=False,
                vivado_path=vivado_path,
            )

            if wns >= 0.000:
                adjusted_high_period_ns = test_period_ns
                best_synthesizable_period = test_period_ns
                best_wns = wns
            else:
                adjusted_low_period_ns = test_period_ns

            print(f"-> Period: {test_period_ns} & WNS: {wns}")

        # Final benchmark implementation run updating README.md.
        final_sweep_wns = synthesis_runner.run_synthesis(
            target_period_ns=best_synthesizable_period,
            should_update_readme=True,
            vivado_path=vivado_path,
        )

        print()
        print(f" --> Best (Shortest) Period: {best_synthesizable_period} ns | Best (Lowest) WNS: {best_wns} ns")

    # Single Out-of-Context synthesis target run.
    elif args.synth is not None:
        target_period: float = args.synth
        print()
        print(f"Running single Out-of-Context synthesis at target period: {target_period} ns...")

        vivado_path = executable_locater.locate_vivado_executable()

        synth_wns = synthesis_runner.run_synthesis(
            target_period_ns=target_period,
            should_update_readme=True,
            vivado_path=vivado_path,
        )