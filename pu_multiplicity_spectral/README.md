# Pu passive neutron multiplicity: shift register vs spectral / bispectrum assay

Simulation-to-assay pipeline. OpenMC transports single source neutrons from a 300 g Pu sphere
(94/6 wt% Pu-239/240) inside a 3-bank He-3/HDPE well counter. The resulting capture library
is stitched into a 300 s list-mode pulse train. Mass, M and α are then recovered by:

* **03** a digital multiplicity shift register and the Böhnel point-model cubic inversion
* **04a** DFT deconvolution of the shift-register histograms, p = IFFT(f̃/b̃), plus a CGF cross-check
* **04b** the PSD and 3-bank cross-PSD (Doubles), and the 3-bank cross-bispectrum (Triples)

## Run

```bash
PY=/home/michel/anaconda3/envs/openmc-env/bin/python   # needs OPENMC_CROSS_SECTIONS (ENDF/B-VIII.1)
$PY 00_env_and_api_check/check_openmc_api.py && $PY 00_env_and_api_check/smoke_test_00.py
$PY 01_openmc_transport/smoke_test_01.py   && $PY 01_openmc_transport/run_openmc_transport.py      # ~10 s
$PY 02_pulse_train_synthesis/smoke_test_02.py && $PY 02_pulse_train_synthesis/synthesize_pulse_train.py
$PY 03_traditional_shift_register/smoke_test_03.py        # also writes the full step-03 results
$PY 04_spectral_bispectrum_assay/smoke_test_04.py         # toy check, then 04a + 04b on OpenMC data (~6 min)
$PY 05_comparison_and_validation/compare_all_methods.py
```

Shared code is in `common/pu_model.py`: the geometry, the point model, the reference rates and logging.
Logs go to `logs/` (`CHANGELOG.md` and `step_XX*.log`), plots to `plots/`, and intermediate files to `data/`.

## Modelling choices

* OpenMC is run as-is. Its fixed-source fission is *implicit*: fission sites are banked at every
  Pu collision. This preserves the mean multiplication but not the induced-fission multiplicity.
  The point model therefore uses **effective** ν_i2 and ν_i3, calibrated from the per-history
  factorial moments of the library (`effective_nu`). The nominal moments give +48% mass.
* `parent_id` in `collision_track.h5` is the history index only within one batch, so transport uses 1 batch.
* Captures later than 10 ms (OpenMC delayed neutrons, about 0.1%) are dropped from the library.
* M_MC is the net (out − in) Pu-surface crossing count per source neutron (`surf_source_write`).
  For 300 g plus the moderator it is 1.29, above the spec's 1.08–1.25 estimate, and the check was widened.
* **Ground truth rates**: `reference_rates` returns the exact expected S, D∞ and T∞ of the synthesis,
  computed from library moments. Inverting them returns exactly 300 g.
* Spectral fits use a two-component kernel (1−w)/(1+iωτ1) + w/(1+iωτ2). The spec's single exponential
  biases D∞ by −1.7% and T∞ by −7%, because thermal-return fission chains are not exponential in time.
  The bispectrum has its own free kernel, so it needs no f_t gate-fraction calibration.
  The shift register's iid f_t is about 3% too high for the same reason.
* Bispectrum normalisation: the permutation *mean* divided by p_X·p_Y·p_Z. This equals the spec's
  4.5 × sum over the 6 permutations. The grid is f1 ∈ (0,10] kHz and f2 ∈ [−10,10] kHz.

## Results (`logs/step_05_final_report.log`)

| | Truth | Shift register P=3µs | 2A DFT | 2B PSD + bispectrum |
|---|---|---|---|---|
| D∞ (cps) | 17404 | 17341 ± 40 | 17341 ± 40 | 17352 ± 75 |
| T∞ (cps) | 13835 | 13421 ± 78 (−3.0%) | 13421 ± 78 | 14008 ± 532 (+1.2%) |
| M | 1.292 | 1.281 | 1.281 | 1.299 ± 0.014 |
| α | 0.50 | 0.456 | 0.456 | 0.53 ± 0.08 |
| m_Pu (g) | 300 | 311.3 ± 2.0 (+3.8%) | 311.3 ± 2.0 | 291 ± 14 (−2.9%) |
