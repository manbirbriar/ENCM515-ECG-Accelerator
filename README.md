# ENCM515 ECG Accelerator

This repo models a **sample-by-sample wearable ECG accelerator** with these hardware units:

- `InputBuffer`
- `FilterMacEngine`
- `SquaringUnit`
- `MovingWindowIntegrator`
- `PeakDetector`
- `ControlUnit`

The repo no longer exposes separate `low_pass`, `high_pass`, and `derivative` hardware modules. Those operations are modeled as internal sub-stages of `FilterMacEngine`.

The simulation infrastructure around those blocks is:

- `DataUploader`
- `ClockUnit`
- `HardwareUnit`
- `DataRecorder`

## Architecture Profiles
- `Arm Cortex-M4`: fixed-point / DSP-style integer timing only
- `Arm Cortex-M4F`: same integer timing plus single-precision FPU timing

The cycle-cost tables are based on official Arm Cortex-M4 TRM instruction timing tables, and the power-per-cycle estimate uses the STM32L476xx datasheet run-mode current figure.

## Clock Comparisons
The repo supports frequency sweeps across:

- `1 kHz` demo clock
- `1 MHz` comparison clock
- `80 MHz` STM32L476-class nominal clock

At high clock rates, the datapath spends most cycles idle because ECG samples only arrive at `360 Hz`.

## Run
```bash
python3 setup.py
```

## Test
```bash
python3 -m unittest -v test_streaming_accelerator.py
```
