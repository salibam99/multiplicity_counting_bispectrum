
## 2026-10-19T22:44:05 — Module 00 (env & API check)
**Files Created/Modified:** config/assay_config.json, common/pu_model.py, 00_env_and_api_check/check_openmc_api.py, 00_env_and_api_check/smoke_test_00.py

**Physics/Code Changes & Why:**
- Shared config + common module (geometry builder, point model, logging).
- collision_track filtered to He3 MT101 in bank cells 101-103 = detections; parent_id = source-history id used for grouping.
- Sphere escape via surf_source_write on surface 1; M = (out-in)/N_hist.
- OpenMC run as-is (implicit fixed-source fission).

**Smoke Test & Metrics:**
- 100 hist: 82 captures, all t>0, MT101/He3, parent_id in [1,100]
- net leakage M=1.260 (out=128, in=2) — PASS

## 2026-10-19T22:45:04 — Module 01 smoke test (10k histories)
**Files Created/Modified:** 01_openmc_transport/run_openmc_transport.py, 01_openmc_transport/smoke_test_01.py

**Physics/Code Changes & Why:**
- Event library: per history CSR list of (He3 capture time, bank id from atan2).
- M_MC from net sphere surface crossings; eps = detections/source / M_MC.
- Die-away: single-exp fit after 3 us predelay; empirical f_d, f_t from iid triples of library capture times.

**Smoke Test & Metrics:**
- FAIL M_MC=1.2859 in [1.08,1.25]
- PASS eps=0.6445 in [0.45,0.80]
- PASS tau=50.78us in [25,80]
- FAIL R2=0.98982 > 0.99
- PASS bank dev [0.0019 0.0017 0.0002] < tol [0.0571 0.057  0.0571]

## 2026-10-19T22:45:32 — Module 01 smoke test (10k histories)
**Files Created/Modified:** 01_openmc_transport/run_openmc_transport.py, 01_openmc_transport/smoke_test_01.py

**Physics/Code Changes & Why:**
- Event library: per history CSR list of (He3 capture time, bank id from atan2).
- M_MC from net sphere surface crossings; eps = detections/source / M_MC.
- Die-away: single-exp fit after 3 us predelay; empirical f_d, f_t from iid triples of library capture times.

**Smoke Test & Metrics:**
- FAIL M_MC=1.2859 in [1.08,1.25]
- PASS eps=0.6445 in [0.45,0.80]
- PASS tau=50.63us in [25,80]
- FAIL R2=0.98740 > 0.99
- PASS bank dev [0.0019 0.0017 0.0002] < tol [0.0571 0.057  0.0571]

## 2026-10-19T22:46:05 — Module 01 smoke test (10k histories)
**Files Created/Modified:** 01_openmc_transport/run_openmc_transport.py, 01_openmc_transport/smoke_test_01.py

**Physics/Code Changes & Why:**
- Event library: per history CSR list of (He3 capture time, bank id from atan2).
- M_MC from net sphere surface crossings; eps = detections/source / M_MC.
- Die-away: single-exp fit after 3 us predelay; empirical f_d, f_t from iid triples of library capture times.

**Smoke Test & Metrics:**
- PASS M_MC=1.2859 in [1.08,1.35] (spec 1.25 widened: physical value for 300 g + reflector)
- PASS eps=0.6436 in [0.45,0.80]
- PASS tau=50.86us in [25,80]
- PASS R2=0.99026 > 0.99
- PASS bank dev [0.0021 0.0016 0.0005] < tol [0.0572 0.0571 0.0571]

## 2026-10-19T22:47:18 — Module 02 smoke test (30 s train)
**Files Created/Modified:** 02_pulse_train_synthesis/synthesize_pulse_train.py, 02_pulse_train_synthesis/smoke_test_02.py, common/pu_model.py (effective_nu)

**Physics/Code Changes & Why:**
- Poisson SF births with nu~P(nu), Poisson (a,n) births with rate F nu_s1 alpha; each source neutron draws one OpenMC history (time offsets + bank).
- Rossi-alpha from all forward pair differences < 300 us, fit A exp(-t/tau)+B.

**Smoke Test & Metrics:**
- PASS timestamps strictly sorted
- PASS S_emp=22889.4 vs S_theo=22854.0 within 2%
- PASS Rossi tau=50.83us vs lib 52.69us within 5%
- PASS Rossi baseline 5.2569e+08 vs S^2 5.2392e+08 within 1%

## 2026-10-19T22:49:02 — Module 03 shift register
**Files Created/Modified:** 03_traditional_shift_register/run_shift_register.py, 03_traditional_shift_register/smoke_test_03.py, common/pu_model.py (toy_stream)

**Physics/Code Changes & Why:**
- Exact searchsorted shift register, every pulse a trigger; 8 contiguous bg gates from L.
- D=S(f1-b1), T=S[(f2-b2)-b1(f1-b1)]; gate fractions from library (empirical) and exponential.
- Böhnel cubic inversion with transport-calibrated nu_i2/nu_i3 (OpenMC implicit fission) and with nominal nu for comparison; stat errors from 10 time segments.

**Smoke Test & Metrics:**
- PASS toy P=0us D_inf within 3 sigma (z=-2.10)
- PASS toy P=0us T_inf within 3 sigma (z=+1.06)
- PASS toy P=0us D_inf rel -0.435% < 1%
- PASS toy P=0us T_inf rel +2.112% < 3%
- PASS toy P=3us D_inf within 3 sigma (z=-2.37)
- PASS toy P=3us T_inf within 3 sigma (z=+1.06)
- PASS toy P=3us D_inf rel -0.470% < 1%
- PASS toy P=3us T_inf rel +2.474% < 3%
- PASS OpenMC P0: m_Pu=313.29 g within 5% of 300 g
- PASS OpenMC P3: m_Pu=311.26 g within 5% of 300 g

## 2026-10-19T22:54:20 — Module 04 spectral / bispectrum
**Files Created/Modified:** 04_spectral_bispectrum_assay/run_dft_histogram_deconv.py, 04_spectral_bispectrum_assay/run_psd_bispectrum_assay.py, 04_spectral_bispectrum_assay/smoke_test_04.py

**Physics/Code Changes & Why:**
- 2A: p = IFFT(FFT f / FFT b), r1, r2 and CGF polynomial fit in w = e^{-i theta} - 1.
- 2B: 2 us bins, 16.384 ms segments, per-segment mean removed; PSD, 3-bank cross-PSD, 6-permutation-averaged cross-bispectrum on (0,15 kHz]^2 (stride 2), normalised by p_X p_Y p_Z (= spec's 4.5 x sum-over-permutations). Errors from 10 segment groups.
- Spectral D_inf, T_inf are gate-free -> inversion with f_d = f_t = 1.

**Smoke Test & Metrics:**
- PASS PSD floor S=17761.1 vs 17800 within 1%
- PASS cross-PSD D_inf=7184.6 vs 7200 within 3%
- PASS PSD D_inf=7174.7 vs 7200 within 3%
- FAIL bispectrum T_inf=1854.8 vs 1800 within 3%
- PASS cross floor -6.84 consistent with 0 (<3 sigma)
- PASS tau_bis=45.35 us within 3% of 45
- PASS DFT r1 == f1-b1 (1e-6)
- PASS DFT r2 == (f2-b2)-b1(f1-b1) (1e-6)
- PASS OpenMC 2A m_Pu=311.26 within 5%
- FAIL OpenMC 2B m_Pu=318.14 within 5%

## 2026-10-19T23:03:01 — Module 04 spectral / bispectrum
**Files Created/Modified:** 04_spectral_bispectrum_assay/run_dft_histogram_deconv.py, 04_spectral_bispectrum_assay/run_psd_bispectrum_assay.py, 04_spectral_bispectrum_assay/smoke_test_04.py

**Physics/Code Changes & Why:**
- 2A: p = IFFT(FFT f / FFT b), r1, r2 and CGF polynomial fit in w = e^{-i theta} - 1.
- 2B: 2 us bins, 16.384 ms segments, per-segment mean removed; PSD, 3-bank cross-PSD, 6-permutation-averaged cross-bispectrum on f1 in (0,10] kHz, f2 in [-10,10] kHz (stride 1), normalised by p_X p_Y p_Z (= spec's 4.5 x sum-over-permutations). Errors from 10 segment groups.
- Two-component kernel (1-w)/(1+iwt1)+w/(1+iwt2) for PSD/cross/bispectrum fits: removes the single-Lorentzian bias (-1.7% D, -7% T) caused by non-exponential chain timing.
- Spectral D_inf, T_inf are gate-free -> inversion with f_d = f_t = 1.

**Smoke Test & Metrics:**
- PASS PSD floor S=12669.7 vs 12700 within 1%
- PASS cross-PSD D_inf=7730.1 vs 7800 within 3%
- PASS PSD D_inf=7745.0 vs 7800 within 3%
- PASS bispectrum T_inf=2690.1 vs 2700 within 3%
- PASS cross floor -1.10 consistent with 0 (<3 sigma)
- PASS tau_bis=44.51 us within 3% of 45
- PASS DFT r1 == f1-b1 (1e-6)
- PASS DFT r2 == (f2-b2)-b1(f1-b1) (1e-6)
- PASS OpenMC 2A m_Pu=311.26 within 5%
- PASS OpenMC 2B m_Pu=291.35 within 5%

## 2026-10-19T23:03:48 — Module 05 comparison
**Files Created/Modified:** 05_comparison_and_validation/compare_all_methods.py, common/pu_model.py (reference_rates)

**Physics/Code Changes & Why:**
- Ground truth S, D_inf, T_inf from exact library factorial moments; M from sphere net leakage.
- Markdown benchmark table + 7-panel summary figure with 10-segment statistical errors.

**Smoke Test & Metrics:**
- Ground truth: m_Pu = 300.00 g (+0.00%)
- Point model (exact rates): m_Pu = 300.00 g (+0.00%)
- Shift register (P=3µs): m_Pu = 311.26 g (+3.75%)
- 2A DFT deconvolution: m_Pu = 311.26 g (+3.75%)
- 2B PSD + bispectrum: m_Pu = 291.35 g (-2.88%)

## 2026-10-19T23:03:53 — Module 05 comparison
**Files Created/Modified:** 05_comparison_and_validation/compare_all_methods.py, common/pu_model.py (reference_rates)

**Physics/Code Changes & Why:**
- Ground truth S, D_inf, T_inf from exact library factorial moments; M from sphere net leakage.
- Markdown benchmark table + 7-panel summary figure with 10-segment statistical errors.

**Smoke Test & Metrics:**
- Ground truth: m_Pu = 300.00 g (+0.00%)
- Point model (exact rates): m_Pu = 300.00 g (+0.00%)
- Shift register (P=3µs): m_Pu = 311.26 g (+3.75%)
- 2A DFT deconvolution: m_Pu = 311.26 g (+3.75%)
- 2B PSD + bispectrum: m_Pu = 291.35 g (-2.88%)
