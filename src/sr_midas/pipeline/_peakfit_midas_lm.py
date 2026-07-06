"""midas_lm — SR-patch peak fitting via the REAL MIDAS-packages fitting engine.

This is the default SR peak-fit routine. Unlike the ``gpu_adam`` /
``gpu_midas_style`` routines (which re-implement a pseudo-Voigt fit in this
package), ``midas_lm`` calls ``midas_peakfit.lm.lm_solve`` directly — the exact
bounded Levenberg-Marquardt + factored pseudo-Voigt model + analytical Jacobian
that MIDAS's ``peakfit_torch`` uses on a plain (non-super-resolved) zarr. So the
peak-fitting *algorithm and equation are identical* to the MIDAS packages
no-SR path; only the input is the x8 super-resolved patch, and the standard SR
coordinate/intensity adjustments are applied around the fit.

What is identical to the MIDAS packages:
  - the fit itself: ``midas_peakfit.lm.lm_solve`` (same LM, same pseudo-Voigt
    ``bg + Σ Imax·[μ·L + (1-μ)·G]`` with factored G/L widths in R and η).
  - the seed formulas: ``midas_peakfit.seeds.seed_region`` (bg=thr/2 ∈ [0,thr];
    Imax ∈ [v/2, 5v]; R±1; Eta±dEta; Mu 0.5 ∈ [0,1]; σ from Voronoi-partitioned
    2nd moments, clipped to [0.1, maxRWidth]).
  - the region-based pixel domain (fit the thresholded region pixels, as the
    packages do — not a zero-padded box).

SR-specific (as in the rest of sr-midas): sub-pixel shift, R/η→(YCen,ZCen)
mapping, Σ-over-SR-grid integrated intensity, srfac×srfac sum-pooled native IMax.

Drop-in signature identical to ``gpu_fit_frame_patches`` so ``sr_process``
dispatches to it via ``peak_fit_method="midas_lm"``.
"""
from __future__ import annotations

import os
import numpy as np
import torch
import torch.nn.functional as F

# Grids + peak detection are shared with the other GPU routines (same package).
from sr_midas.pipeline._gpu_peakfit import build_RE_grids, detect_peaks_and_init

_RAD2DEG = 57.29577951308232
_DEG2RAD = 0.017453292519943295

# Patches per lm_solve call; region pixels are ~10-50x fewer than the full patch
# so we can fit many patches at once.
_MAX_BATCH = int(os.environ.get("MIDAS_LM_MAX_BATCH", "512"))
# Padded region size per patch (top-N brightest region pixels; generous — peaks
# are localized). Region = SR pixels above the SR-scale ring threshold.
_MAX_REGION_PX = int(os.environ.get("MIDAS_LM_MAX_REGION_PX", "4096"))


def _require_midas_peakfit():
    try:
        from midas_peakfit.lm import lm_solve, LMConfig  # noqa: F401
        return lm_solve, LMConfig
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "peak_fit_method='midas_lm' requires the 'midas-peakfit' package "
            "(pip install midas-peakfit, or midas-suite[sr]). Original error: "
            f"{exc}"
        ) from exc


def _render_pv_per_peak_midas(Rs, Etas, params_bk_k8):
    """Per-peak factored pseudo-Voigt (no bg), MIDAS layout
    (Imax,R,Eta,Mu,sigGR,sigLR,sigGEta,sigLEta). Rs,Etas: (B,M) -> (B,k,M).
    Matches midas_peakfit.model.forward_pseudo_voigt term-for-term."""
    Imax = params_bk_k8[..., 0].unsqueeze(-1)
    R0 = params_bk_k8[..., 1].unsqueeze(-1); Eta0 = params_bk_k8[..., 2].unsqueeze(-1)
    Mu = params_bk_k8[..., 3].unsqueeze(-1)
    sGR = params_bk_k8[..., 4].unsqueeze(-1); sLR = params_bk_k8[..., 5].unsqueeze(-1)
    sGE = params_bk_k8[..., 6].unsqueeze(-1); sLE = params_bk_k8[..., 7].unsqueeze(-1)
    RR = Rs.unsqueeze(1); EE = Etas.unsqueeze(1)
    dRG = (RR - R0) / sGR; dEG = (EE - Eta0) / sGE
    G = torch.exp(-0.5 * dRG * dRG - 0.5 * dEG * dEG)
    dRL = (RR - R0) / sLR; dEL = (EE - Eta0) / sLE
    L = 1.0 / ((1.0 + dRL * dRL) * (1.0 + dEL * dEL))
    return Imax * (Mu * L + (1.0 - Mu) * G)


def _seed_group(peak_R, peak_E, peak_I, Rs, Etas, z, thr_native, srfac, dtype, device):
    """Build MIDAS-style x0/xl/xu for a batch of patches with exactly k peaks.
    Replicates midas_peakfit.seeds.seed_region formulas on the SR grid."""
    B, k = peak_R.shape
    M = Rs.shape[1]
    thr_sr = float(thr_native) / float(srfac * srfac)

    valid = z > 0
    big = torch.tensor(1e9, dtype=dtype, device=device)
    Rmin = torch.where(valid, Rs, big).amin(dim=1)
    Rmax = torch.where(valid, Rs, -big).amax(dim=1)
    Emin = torch.where(valid, Etas, big).amin(dim=1)
    Emax = torch.where(valid, Etas, -big).amax(dim=1)
    empty = ~valid.any(dim=1)
    Rmin = torch.where(empty, peak_R.amin(dim=1), Rmin)
    Rmax = torch.where(empty, peak_R.amax(dim=1), Rmax)
    Emin = torch.where(empty, peak_E.amin(dim=1), Emin)
    Emax = torch.where(empty, peak_E.amax(dim=1), Emax)

    maxRWidth = (Rmax - Rmin) / 2.0 + 1.0
    denom = (Rmax + Rmin)
    atan_term = torch.atan(torch.where(denom != 0, 2.0 / denom, torch.zeros_like(denom))) * _RAD2DEG
    maxEtaWidth = (Emax - Emin) / 2.0 + atan_term
    maxEtaWidth = torch.where((Emax - Emin) > 180.0, maxEtaWidth - 180.0, maxEtaWidth)
    maxRWidth = torch.clamp(maxRWidth, min=0.1)
    maxEtaWidth = torch.clamp(maxEtaWidth, min=0.1)

    width = torch.sqrt(torch.tensor(float(M), dtype=dtype, device=device) / max(k, 1))
    width = torch.minimum(width.expand(B), maxRWidth)

    val = z - thr_sr / 2.0
    pos = val > 0
    valp = torch.where(pos, val, torch.zeros_like(val))
    dR = Rs.unsqueeze(2) - peak_R.unsqueeze(1)
    dE = Etas.unsqueeze(2) - peak_E.unsqueeze(1)
    d2 = dR * dR + dE * dE
    closest = torch.argmin(d2, dim=2)
    dR_c = torch.gather(dR, 2, closest.unsqueeze(2)).squeeze(2)
    dE_c = torch.gather(dE, 2, closest.unsqueeze(2)).squeeze(2)
    sumW = torch.zeros(B, k, dtype=dtype, device=device)
    sumWR2 = torch.zeros(B, k, dtype=dtype, device=device)
    sumWE2 = torch.zeros(B, k, dtype=dtype, device=device)
    sumW.scatter_add_(1, closest, valp)
    sumWR2.scatter_add_(1, closest, valp * dR_c * dR_c)
    sumWE2.scatter_add_(1, closest, valp * dE_c * dE_c)
    ok = sumW > 0
    sR = torch.sqrt(torch.where(ok, sumWR2 / torch.where(ok, sumW, torch.ones_like(sumW)), torch.zeros_like(sumW)))
    sE = torch.sqrt(torch.where(ok, sumWE2 / torch.where(ok, sumW, torch.ones_like(sumW)), torch.zeros_like(sumW)))
    estimSigmaR = torch.where(ok, torch.clamp(torch.minimum(sR, maxRWidth.unsqueeze(1)), min=0.1), width.unsqueeze(1).expand(B, k))
    estimSigmaEta = torch.where(ok, torch.clamp(sE, min=0.1), width.unsqueeze(1).expand(B, k))
    dEta = _RAD2DEG * torch.atan(1.0 / torch.clamp(peak_R, min=1e-9))

    n = 1 + 8 * k
    x0 = torch.zeros(B, n, dtype=dtype, device=device)
    xl = torch.zeros(B, n, dtype=dtype, device=device)
    xu = torch.zeros(B, n, dtype=dtype, device=device)
    x0[:, 0] = thr_sr / 2.0; xl[:, 0] = 0.0; xu[:, 0] = thr_sr
    for i in range(k):
        b = 8 * i
        x0[:, b + 1] = peak_I[:, i]; x0[:, b + 2] = peak_R[:, i]; x0[:, b + 3] = peak_E[:, i]
        x0[:, b + 4] = 0.5
        x0[:, b + 5] = estimSigmaR[:, i]; x0[:, b + 6] = estimSigmaR[:, i]
        x0[:, b + 7] = estimSigmaEta[:, i]; x0[:, b + 8] = estimSigmaEta[:, i]
        xl[:, b + 1] = peak_I[:, i] / 2.0; xl[:, b + 2] = peak_R[:, i] - 1.0
        xl[:, b + 3] = peak_E[:, i] - dEta[:, i]; xl[:, b + 4] = 0.0
        xl[:, b + 5] = 0.01; xl[:, b + 6] = 0.01; xl[:, b + 7] = 0.005; xl[:, b + 8] = 0.005
        xu[:, b + 1] = peak_I[:, i] * 5.0; xu[:, b + 2] = peak_R[:, i] + 1.0
        xu[:, b + 3] = peak_E[:, i] + dEta[:, i]; xu[:, b + 4] = 1.0
        xu[:, b + 5] = 2.0 * maxRWidth; xu[:, b + 6] = 2.0 * maxRWidth
        xu[:, b + 7] = 2.0 * maxEtaWidth; xu[:, b + 8] = 2.0 * maxEtaWidth
    xu[:, 1::8][:, :k] = torch.clamp(xu[:, 1::8][:, :k], min=1e-6)
    x0 = torch.clamp(x0, xl, xu)
    return x0, xl, xu


def midas_lm_fit_frame_patches(patches_to_fit_t, patches_Y00, patches_Z00,
                               patches_exp_t, nr_pixels_in_patch, patches_Isum,
                               sr_params, sr_config, srfac,
                               omega, shiftYpx, shiftZpx, torch_devs,
                               n_steps=20, lr=0.15, use_compile=True, logger=None):
    """Fit one frame's SR patches with the real midas_peakfit LM engine.
    Returns ``(df_rows, n_peaks_list, spotID)`` (29-col MIDAS schema)."""
    lm_solve, LMConfig = _require_midas_peakfit()
    n_patches = int(patches_to_fit_t.shape[0])
    if n_patches == 0:
        return [], [], 0

    device = torch_devs
    dtype = torch.float32
    lrsz = int(sr_config["lrsz"])
    Ypx_BC = float(sr_params["Ypx_BC"]); Zpx_BC = float(sr_params["Zpx_BC"])
    lr_int_thresh = sr_config["peak_find_args"]["pvfit_int_thresh"][f"SRx{srfac}"]
    min_d = sr_config["peak_find_args"]["min_d"][f"SRx{srfac}"]
    thresh_rel = sr_config["peak_find_args"]["thresh_rel"][f"SRx{srfac}"]
    rings_thresh = sr_params.get("ringsThresh", None)
    thr_native = float(np.median(np.asarray(rings_thresh, dtype=np.float64))) \
        if (rings_thresh is not None and len(rings_thresh) > 0) else float(lr_int_thresh)
    thr_sr = thr_native / float(srfac * srfac)

    patches_t = patches_to_fit_t[:, 0].contiguous().to(device=device, dtype=dtype)

    def _tf(x):
        if isinstance(x, torch.Tensor):
            return x.to(device=device, dtype=dtype)
        return torch.as_tensor(np.asarray(x, dtype=np.float32), device=device)

    Y00_t = _tf(patches_Y00); Z00_t = _tf(patches_Z00)
    isum_t = _tf(patches_Isum); nrpx_t = _tf(nr_pixels_in_patch)
    exp_t = patches_exp_t.to(device=device, dtype=dtype)

    grid_RR, grid_EE = build_RE_grids(Y00_t, Z00_t, lrsz, srfac, Ypx_BC, Zpx_BC, device)
    grid_RR = grid_RR.to(dtype); grid_EE = grid_EE.to(dtype)
    init_sr, n_peaks_detected, _lb, _ub = detect_peaks_and_init(
        patches_t, grid_RR, grid_EE, srfac,
        min_distance=min_d, threshold_rel=thresh_rel,
        edge_bound_cutoff_fac=0.0, exclude_border=True)
    peak_R_all = init_sr[..., 0]; peak_E_all = init_sr[..., 1]; peak_I_all = init_sr[..., 7]

    H = W = lrsz * srfac
    M_full = H * W
    z_full = patches_t.reshape(n_patches, M_full)
    Rs_full = grid_RR.reshape(n_patches, M_full)
    Etas_full = grid_EE.reshape(n_patches, M_full)
    region_mask = z_full > thr_sr
    region_count = region_mask.sum(dim=1)

    lmcfg = LMConfig(max_iter=100, ftol_rel=1e-4, xtol_rel=1e-4,
                     lambda_init=1e-3, use_torch_compile=bool(use_compile))
    n_peaks_np = n_peaks_detected.detach().cpu().numpy().astype(int)
    n_peaks_list = [int(v) for v in n_peaks_np]
    out_rows = []

    kmax = int(n_peaks_np.max()) if n_patches else 1
    for k in range(1, kmax + 1):
        idx_k = np.nonzero(n_peaks_np == k)[0]
        if idx_k.size == 0:
            continue
        idx_k_t = torch.as_tensor(idx_k, device=device, dtype=torch.long)
        for s in range(0, idx_k.size, _MAX_BATCH):
            sub = idx_k_t[s:s + _MAX_BATCH]
            B = int(sub.shape[0])
            rc_mask = region_mask.index_select(0, sub)
            rcount = region_count.index_select(0, sub)
            Mreg = int(min(int(rcount.max().item()), _MAX_REGION_PX)) or 1
            zb = z_full.index_select(0, sub); Rb = Rs_full.index_select(0, sub); Eb = Etas_full.index_select(0, sub)
            score = torch.where(rc_mask, zb, torch.full_like(zb, -1.0))
            topv, topi = torch.topk(score, k=Mreg, dim=1)
            valid = topv > -1.0
            z = torch.gather(zb, 1, topi); Rs = torch.gather(Rb, 1, topi); Etas = torch.gather(Eb, 1, topi)
            pmask = valid.to(dtype)
            pR = peak_R_all.index_select(0, sub)[:, :k].contiguous()
            pE = peak_E_all.index_select(0, sub)[:, :k].contiguous()
            pI = torch.clamp(peak_I_all.index_select(0, sub)[:, :k].contiguous(), min=1e-6)
            x0, xl, xu = _seed_group(pR, pE, pI, Rs, Etas, z, thr_native, srfac, dtype, device)
            x_final, cost, rc, _sig = lm_solve(x0, xl, xu, z, Rs, Etas, pmask, n_peaks=k, config=lmcfg)

            per_peak = x_final[:, 1:].reshape(B, k, 8)
            Imax = per_peak[..., 0]; Rv = per_peak[..., 1]; Ev = per_peak[..., 2]; Mu = per_peak[..., 3]
            sGR = per_peak[..., 4]; sLR = per_peak[..., 5]; sGE = per_peak[..., 6]; sLE = per_peak[..., 7]
            SigmaR = torch.maximum(sGR, sLR); SigmaEta = torch.maximum(sGE, sLE)
            bg = x_final[:, 0]
            eta_rad = Ev * _DEG2RAD
            YCen = Ypx_BC + Rv * torch.sin(eta_rad) + float(shiftYpx)
            ZCen = Zpx_BC + Rv * torch.cos(eta_rad) + float(shiftZpx)
            Rs_g = Rs_full.index_select(0, sub); Etas_g = Etas_full.index_select(0, sub)
            pv = _render_pv_per_peak_midas(Rs_g, Etas_g, per_peak)
            integrated = pv.sum(dim=2)
            pv_img = pv.reshape(B, k, H, W)
            if srfac > 1:
                pooled = F.avg_pool2d(pv_img.reshape(B * k, 1, H, W), kernel_size=srfac, stride=srfac, divisor_override=1)
                pv_srx1 = pooled.reshape(B, k, H // srfac, W // srfac)
            else:
                pv_srx1 = pv_img
            Ws = pv_srx1.shape[-1]
            imax_out = pv_srx1.amax(dim=(-2, -1))
            argmax_flat = pv_srx1.flatten(start_dim=-2).argmax(dim=-1)
            r_max = (argmax_flat // Ws).to(dtype); c_max = (argmax_flat % Ws).to(dtype)
            Y00_sub = Y00_t.index_select(0, sub).unsqueeze(1); Z00_sub = Z00_t.index_select(0, sub).unsqueeze(1)
            maxY = Y00_sub + c_max; maxZ = Z00_sub + r_max
            exp_sub = exp_t.index_select(0, sub).squeeze(1).unsqueeze(1)
            nr_pixels = ((pv_srx1 * exp_sub) != 0).sum(dim=(-2, -1)).to(dtype)
            diffY = maxY - YCen; diffZ = maxZ - ZCen
            fit_rmse = torch.sqrt(torch.clamp(cost / rcount.clamp(min=1), min=0.0)).unsqueeze(1).expand(B, k)
            rc_f = rc.to(dtype).unsqueeze(1).expand(B, k)
            omega_t = torch.full((B, k), float(omega), dtype=dtype, device=device)
            nPeaks_t = torch.full((B, k), float(k), dtype=dtype, device=device)
            total_nrpx = nrpx_t.index_select(0, sub).unsqueeze(1).expand(B, k)
            rawIMax = exp_t.index_select(0, sub).amax(dim=(-2, -1)).squeeze(1).unsqueeze(1).expand(B, k)
            raw_isum = isum_t.index_select(0, sub).unsqueeze(1).expand(B, k)
            bg_bk = bg.unsqueeze(1).expand(B, k)
            zeros_bk = torch.zeros(B, k, dtype=dtype, device=device)
            cols = [zeros_bk, integrated, omega_t, YCen, ZCen, imax_out, Rv, Ev, SigmaR, SigmaEta,
                    nr_pixels, total_nrpx, nPeaks_t, maxY, maxZ, diffY, diffZ, rawIMax, rc_f,
                    fit_rmse, bg_bk, sGR, sLR, sGE, sLE, Mu, raw_isum, zeros_bk, fit_rmse]
            rows_bk = torch.stack(cols, dim=-1)
            rows_np = rows_bk.detach().cpu().numpy()
            sub_np = sub.detach().cpu().numpy()
            for bi in range(B):
                pidx = int(sub_np[bi])
                for pj in range(k):
                    out_rows.append((pidx, pj, rows_np[bi, pj].tolist()))

    out_rows.sort(key=lambda t: (t[0], t[1]))
    df_rows = []
    for spot_i, (_p, _j, row) in enumerate(out_rows, start=1):
        row[0] = float(spot_i)
        df_rows.append(row)
    return df_rows, n_peaks_list, len(df_rows)
