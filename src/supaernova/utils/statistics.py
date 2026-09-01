import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from supaernova.configs.steps.data import SNPAEData, LazySNPAEData
    from numpy import typing as npt
    from collections.abc import Callable


def SNR(
    data: "SNPAEData | LazySNPAEData",
    mask: "npt.NDArray | None" = None,
    normalise: "bool" = False,
    reduce: "Callable | None" = None,
):
    amplitude = np.clip(data.amplitude, 0, np.inf)
    sigma = data.sigma
    if mask is None:
        mask = np.ones_like(data.mask, dtype=bool)
    else:
        mask = mask.astype(bool)

    mask = (
        mask
        & (data.mask != 0)
        & (data.sn_mask != 0)
        & (data.spec_mask != 0)
        & (data.wl_mask != 0)
    )
    if np.count_nonzero(mask) == 0:
        return 0

    signal = np.where(mask, amplitude * amplitude, 0.0)
    noise = np.where(mask, sigma * sigma, 0.0)

    # Compute division only where noise > 0 to avoid 0 / 0 runtime warnings
    snr = np.zeros_like(signal, dtype=float)
    valid_noise = mask & (noise > 0)
    np.divide(signal, noise, out=snr, where=valid_noise)
    snr = np.where(np.isfinite(snr), snr, 0.0)

    snr_sum = np.sum(snr, axis=(-2, -1))
    if normalise:
        counts = np.count_nonzero(mask, axis=(-2, -1))
        valid_counts = counts > 0
        np.divide(snr_sum, counts, out=snr_sum, where=valid_counts)

    snr_mask = np.isfinite(snr_sum)
    snr_sum = np.where(snr_mask, snr_sum, 0.0)
    snr_coadd = np.sqrt(snr_sum)

    valid_snrs = np.count_nonzero(snr_mask)
    if reduce is None:
        return np.sum(snr_coadd) / valid_snrs if valid_snrs > 0 else 0.0
    return reduce(np.where(snr_mask, snr_coadd, 0.0))
