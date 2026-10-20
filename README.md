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
6. **Studies** (`06`), to test whether the new method really has a case:
   - 8 repeated 300 s measurements, to see bias and spread separately;
   - a realistic **Cf-252 calibration** of the shift register's gate fractions;
   - a **moderated item**: the same Pu sphere inside a 1.5 cm polyethylene shell;
   - an (α,n) rate scan;
   - a **hybrid**, in which the bispectrum calibrates the shift-register gate fraction f_t.

Every step has a smoke test; logs are in `pu_multiplicity_spectral/logs/` and plots in
`pu_multiplicity_spectral/plots/`. See `pu_multiplicity_spectral/README.md` for how to run it.
The full write-up (theory, geometry, figures, studies) is `documentation/report.pdf`.

**Caveat:** OpenMC treats fixed-source fission implicitly. That preserves the average
multiplication but not the spread of neutrons per induced fission. Both methods therefore use
effective induced-fission moments calibrated from the transport data. This is an approximation
in the simulated physics; it does not affect the comparison between the two methods.

## Conclusion: legacy vs new method

Repeated 300 s measurements; true mass 300 g. The bias is on the mean mass, the spread is the
1σ error of a single run, and the RMS combines both.

| Item | Method | Mass bias | Spread / run | RMS error |
|---|---|---|---|---|
| Bare Pu sphere (8 runs) | Legacy: shift register, Cf-252 calibrated | +4.5% | 2.1 g | 13.7 g |
| | New: cross-PSD + bispectrum | +0.8 ± 1.1% | 9.5 g | 9.2 g |
| | **Hybrid** (bispectrum sets f_t) | +0.8 ± 0.3% | 2.8 g | **3.5 g** |
| Moderated item (4 runs) | Legacy: shift register, Cf-252 calibrated | +5.3% | 1.4 g | 15.8 g |
| | New: cross-PSD + bispectrum | −0.7 ± 1.3% | 7.5 g | 6.8 g |
| | **Hybrid** (bispectrum sets f_t) | −0.7 ± 0.5% | 2.8 g | **3.3 g** |

- **Legacy method: precise but biased, and calibration doesn't fix it.** A Cf-252 fission is
  a single burst. In a Pu item, neutrons also return from the polyethylene and cause later
  fissions, which spread the triplets out in time. So the Cf-calibrated Triples gate fraction
  is 4% too high for the bare item and 8% too high for the moderated one. This is a matrix
  effect the calibration source cannot see.
- **New method: unbiased on both items, with no gate fractions and no calibration source.**
  Rates are read at zero frequency, where the detector response equals 1, and the background
  cancels by construction. The cost is precision: 2–4× worse on the Triples rate, with no
  gain at high (α,n) rates. Even so, its total error for a single run is already smaller
  than that of the calibrated shift register.
- **Recommended: the hybrid.** The bispectrum measures f_t for the item type, from other
  measurements of the same kind of item (here about 35 min), and the shift register supplies
  the precision. Result: about 3 g RMS on 300 g (1%).
- **Still to confirm:** the size of the chain-timing effect depends on OpenMC's implicit
  fission treatment. It should be checked with an analog fission code (e.g. MCNPX-PoliMi) or
  with measurements.
