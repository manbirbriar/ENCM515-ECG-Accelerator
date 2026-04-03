from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class OperationCycleTable:
  add: int
  sub: int
  mul: int
  mac: int
  shift: int
  compare: int


@dataclass(frozen=True, slots=True)
class CoreProfile:
  name: str
  label: str
  fixed_cycles: OperationCycleTable
  float_cycles: OperationCycleTable | None
  hardware_float_supported: bool
  notes: str


FIXED_POINT_SCALE = 2**15 - 1
FIXED_POINT_BITS = 15

# MIT-BIH sample rate.
SAMPLE_RATE_HZ: int = 360
SAMPLE_RATE: int = SAMPLE_RATE_HZ

# Sample-by-sample streaming model.
PROCESSING_WINDOW_SIZE: int = 1

# Cortex-M4/M4F clock comparison points. 80 MHz matches STM32L476 nominal max clock.
DEMO_CLOCK_HZ: int = 1_000
LOW_CLOCK_HZ: int = 1_000_000
STM32L476_MAX_CLOCK_HZ: int = 80_000_000
DEFAULT_ACCEL_CLOCK_HZ: int = LOW_CLOCK_HZ
CLOCK_SWEEP_HZ: tuple[int, ...] = (DEMO_CLOCK_HZ, LOW_CLOCK_HZ, STM32L476_MAX_CLOCK_HZ)

# Streaming buffer / recorder sizing.
INGRESS_FIFO_DEPTH: int = 512
DATA_RECORDER_CAPACITY: int = 2500

# Moving-window integration and refractory period.
MWI_WINDOW_SIZE: int = int(SAMPLE_RATE_HZ * 0.15)
REFRACTORY_PERIOD_SAMPLES: int = int(SAMPLE_RATE_HZ * 0.2)
# Ignore early filter transients before enabling adaptive peak detection.
PEAK_DETECTOR_STARTUP_SAMPLES: int = int(SAMPLE_RATE_HZ * 0.4)

# Reference power model:
# STM32L476xx datasheet Rev 11 (July 2024), feature summary:
# 39 uA/MHz run mode at 3.3 V in SMPS mode.
REFERENCE_DEVICE_NAME: str = "STM32L476xx"
REFERENCE_SUPPLY_VOLTAGE_V: float = 3.3
REFERENCE_RUN_CURRENT_UA_PER_MHZ: float = 39.0
REFERENCE_ENERGY_PER_CYCLE_J: float = (
  REFERENCE_RUN_CURRENT_UA_PER_MHZ * 1e-6 * REFERENCE_SUPPLY_VOLTAGE_V / 1e6
)

# Arm Cortex-M4 core timing references:
# - Arm Cortex-M4 TRM Table 3-1: ADD, SUB, CMP, LSL = 1 cycle.
# - Arm Cortex-M4 community blog / DSP datasheet: MUL and MAC are single-cycle.
# - Arm Cortex-M4 TRM Table 7-1: Cortex-M4F VADD, VSUB, VMUL, VCMP = 1 cycle.
# - For plain Cortex-M4 without FPU, GCC/Arm toolchains use runtime-library software
#   emulation for floating-point operations when built with soft-float options.
#   Those costs are toolchain-dependent, so the values below are a modeling estimate
#   chosen to be much slower than fixed-point and to reflect library-call overhead.
#   shift unused in float datapath
CORTEX_M4_PROFILE = CoreProfile(
  name="cortex_m4",
  label="Arm Cortex-M4",
  fixed_cycles=OperationCycleTable(add=1, sub=1, mul=1, mac=1, shift=1, compare=1),
  float_cycles=OperationCycleTable(add=25, sub=25, mul=35, mac=60, shift=1, compare=15),
  hardware_float_supported=False,
  notes=(
    "No hardware FPU. Float timing is modeled as software-emulated runtime-library operations, "
    "so it is intentionally much slower than fixed-point."
  ),
)

CORTEX_M4F_PROFILE = CoreProfile(
  name="cortex_m4f",
  label="Arm Cortex-M4F",
  fixed_cycles=OperationCycleTable(add=1, sub=1, mul=1, mac=1, shift=1, compare=1),
  float_cycles=OperationCycleTable(add=1, sub=1, mul=1, mac=1, shift=1, compare=1),
  hardware_float_supported=True,
  notes="Includes single-precision FPU with 1-cycle VADD/VSUB/VMUL/VCMP instruction timing in the TRM table.",
)

DEFAULT_CORE_PROFILE: CoreProfile = CORTEX_M4F_PROFILE


def get_cycle_table(core_profile: CoreProfile, is_fixed_point: bool) -> OperationCycleTable:
  if is_fixed_point:
    return core_profile.fixed_cycles

  if core_profile.float_cycles is None:
    raise ValueError(
      f"{core_profile.label} does not provide a floating-point timing model."
    )

  return core_profile.float_cycles
