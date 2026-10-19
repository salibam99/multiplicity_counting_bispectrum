# Spectral / Bispectrum Neutron Multiplicity Counting for Pu Assay

## Goal
Passive neutron multiplicity counting estimates an unknown plutonium mass from the time
correlations between detected neutrons. Three rates are measured: Singles (S), Doubles (D)
and Triples (T). The point model then gives the mass, the self-multiplication M and the
(α,n) ratio α.

The **legacy method** is the *shift register*. It counts neutrons in gates after each trigger.
To turn gated counts into true rates it needs calibrated gate fractions f_d and f_t, which
depend on the detector's die-away time profile.

The **new method** works on the pulse train in the **frequency domain**:
- **Doubles** come from the low-frequency plateau of the 3-bank **cross-PSD**. The uncorrelated
  white-noise floor (Singles) drops out.
- **Triples** come from the zero-frequency value of the 3-bank **cross-bispectrum**
  B_xyz(ω1, ω2). Singles and Doubles vanish identically, because only a correlated triplet
  can hit three different detector banks.

The spectral quantities are total rates, so **no gate fractions are needed**. The time kernel
is fitted directly from the spectrum.

## What was done
1. **OpenMC transport** (`01`): a 300 g Pu metal sphere (6% Pu-240) in a 3-bank He-3/HDPE
   well counter. 400k source neutrons are tracked; `collision_track` records each He-3
   capture's time and bank, and surface tracking gives M. Results: ε = 0.643,
   τ = 52.7 µs, M = 1.292.
2. **Pulse train synthesis** (`02`): 300 s of list-mode data. Spontaneous fissions
   (P(ν) for Pu-240) and (α,n) neutrons arrive as Poisson processes, and each neutron draws
   an OpenMC history. That gives about 6.8 M pulses.
3. **Legacy assay** (`03`): an exact digital shift register (P = 3 µs, G = 64 µs) followed by
   the Böhnel cubic inversion.
4. **New assay** (`04`):
   - 2A: DFT deconvolution of the shift-register histograms.
   - 2B: PSD, cross-PSD and cross-bispectrum.
5. **Comparison** (`05`) against the exact expected rates of the synthetic measurement.

Every step has a smoke test; logs are in `pu_multiplicity_spectral/logs/` and plots in
`pu_multiplicity_spectral/plots/`. See `pu_multiplicity_spectral/README.md` for how to run it.

**Caveat:** OpenMC treats fixed-source fission implicitly. That preserves the average
multiplication but not the spread of neutrons per induced fission. Both methods therefore use
effective induced-fission moments calibrated from the transport data. This is an approximation
in the simulated physics; it does not affect the comparison between the two methods.

## Conclusion: legacy vs new method (300 s measurement)

| Quantity | Ground truth | Legacy: shift register | New: cross-PSD + bispectrum |
|---|---|---|---|
| Doubles D∞ (cps) | 17404 | 17341 ± 40 (−0.4%) | 17352 ± 75 (−0.3%) |
| Triples T∞ (cps) | 13835 | 13421 ± 78 (**−3.0%**) | 14008 ± 532 (+1.2%) |
| Multiplication M | 1.292 | 1.281 ± 0.002 | 1.299 ± 0.014 |
| α ratio | 0.50 | 0.456 ± 0.008 (−8.8%) | 0.53 ± 0.08 (+6.9%) |
| **Pu mass (g)** | **300** | **311.3 ± 2.0 (+3.8%)** | **291 ± 14 (−2.9%)** |

- **Legacy method:** precise but **biased**. Its Triples gate fraction assumes a pure
  exponential die-away. The real fission chains have slower time structure, so T∞ comes out
  3% low and the mass about 4% high.
- **New method:** **unbiased within its error**. It needs no gate-fraction calibration, and
  the cross-spectra remove the accidental (Singles) background by construction.
- **Price of the new method:** larger statistical uncertainty on the Triples (±3.8% vs ±0.6%).
  Longer measurements or a constrained kernel fit would improve it.
