# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Single-file 5G NR DSP simulation (`nr_subband_stitch.py`). One OFDM symbol (100MHz carrier, SCS=30kHz,
273 RB → 3276 subcarriers) is built in the frequency domain, IFFT'd into a 4096-sample "ADC time signal,"
then demodulated two ways whose results are compared:

- **Path A (`fullband_demod`)** — direct 4096-FFT, the golden reference.
- **Path B (`subband_demod`)** — models a chip that cannot do a 4096-FFT, split into a mid-RF (IF) and
  baseband (BB) stage over 5 subbands (RB=[55,54,55,54,55]):
  - IF: `DDC → circular LPF → decimate 4× → apply actual_delay` (IF→BB transfer delay, time domain pre-FFT)
  - BB: `1024-FFT → phase comp (FIR group delay + comp_delay) → stitch` into the full 3276.

`main()` runs the whole flow for two test signals (`'qpsk'`, `'ones'`) with matched delays, plus one
mismatched-delay control run, prints max/mean amplitude (dB) and phase (deg) differences, saves
`result_<mode>.png`.

## Commands

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python nr_subband_stitch.py     # prints metrics, writes result_*.png
```

No test suite or linter. The script's own **self-check** (`fullband ref vs input` ≈ 1e-15) is the
correctness gate for the IFFT/FFT/mapping plumbing — if that grows large, the frequency mapping broke.

## Critical invariants (don't break these when editing)

These exact ratios make every FFT bin map to exactly one subcarrier, enabling leakage-free stitching:

- `SCS/FS = 30e3/122.88e6 = 1/NFFT (1/4096)` — DDC by an integer number of subcarriers is an integer
  number of cycles over 4096 samples (circular-friendly).
- `decimated rate / NSUB = 30.72e6/1024 = 30kHz = SCS` — one 1024-FFT bin per subcarrier.
- Subband `center` is **rounded to an integer subcarrier index** (`subband_layout`) so subcarriers land
  on integer 1024-FFT bins after DDC. A fractional center reintroduces inter-bin leakage.

Other non-obvious points:

- **Scaling**: subband path multiplies by `D` to undo the `1/D` amplitude from decimation; the full-band
  path needs no factor because numpy `fft∘ifft` is identity. Verify via the `'ones'` run staying ~0 dB.
- **Phase compensation** is the headline metric. Raw phase is dominated by the FIR linear-phase ramp
  (~180° sawtooth) plus any transfer delay; `subband_demod` returns both `stitched_comp` (compensated)
  and `stitched_raw`. Compensation has two independent terms on signed offset `m = s − center`:
  `exp(+j·2π·m·g/NFFT)` (FIR group delay `g`, full-rate domain) and
  `exp(+j·2π·m·comp_delay/NSUB)` (per-subband transfer delay, decimated domain).
- **Per-subband transfer delay** (`ACTUAL_DELAYS`/`COMP_DELAYS`, decimated samples, fractional ok):
  `actual_delay` is applied via `apply_circular_delay` on the 1024-point decimated signal *before* FFT;
  `comp_delay` is the frequency-domain inverse. Matched → exact cancellation (all `|m| < NSUB/2`, so
  `fftfreq` bin = signed `m` exactly); a Δ mismatch grows phase error as `360·m·Δ/NSUB` deg.
- **Filtering is circular** (`circular_filter`, FFT-based) on purpose — the symbol is treated as periodic
  to remove block-edge transients and isolate pure stitching error. Do not switch to linear `lfilter`
  without re-deriving the delay/edge handling.
- Residual error concentrates at **subband-edge subcarriers** (FIR passband ripple + finite stopband
  leakage). Degrading `design_lpf()` (fewer taps / lower stopband) is the intended knob to amplify it.

## Plots / fonts

Matplotlib uses the `Agg` backend and **English** axis labels deliberately — the default font lacks CJK
glyphs, so keep plot text ASCII (console output is Chinese, which is fine).
