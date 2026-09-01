from typing import TYPE_CHECKING, Any, Literal
from pathlib import Path

import numpy as np
import pandas as pd

from .spectra import SpectraPlot
from .analysis import Plotter

if TYPE_CHECKING:
    from numpy import typing as npt

    from supaernova.configs.steps.steps import StepResult

    from .analysis import Axis, Figure


class DistributionPlot(SpectraPlot):
    labels: "dict[str | int, str | dict[str | int, str]] | None" = None
    mean: bool = False
    masked: bool = False
    reduce: Literal["mean", "median", "max_central"] = "max_central"


class DistributionPlotter(Plotter):
    @staticmethod
    def prep_from_result(data: "StepResult", config: DistributionPlot) -> pd.DataFrame:
        return pd.DataFrame({
            label: getattr(data, str(key))
            for (key, label) in (config.labels or {}).items()
        })

    @staticmethod
    def prep_from_array(
        data: "npt.NDArray[Any]", config: DistributionPlot
    ) -> pd.DataFrame:
        return pd.DataFrame({
            label: data[:, ind]
            for (ind, label) in (config.labels or {}).items()
            if ind < data.shape[-1]
        })

    @staticmethod
    def plot_corner(
        data: "StepResult | npt.NDArray[Any] | list[StepResult] | list[npt.NDArray[Any]] | dict[str, Any]",
        config: "DistributionPlot",
        *,
        fig: "Figure | None" = None,
        ax: "Axis | None" = None,
        force: bool = False,
        save: bool = True,
        **chain_kwargs: Any,
    ) -> tuple["Figure", "Axis"] | tuple[None, None]:
        savepath = (config.savepath or Path()) / f"{config.name}.{config.ext}"
        if savepath.exists() and not force:
            Plotter.close(fig, ax)
            return None, None

        labels = None
        if isinstance(data, dict):
            labels = list(data.keys())
            data = list(data.values())

        if not isinstance(data, list):
            data = [data]

        if config.mean:
            chains = {
                "mean": DistributionPlotter.prep_from_array(
                    np.mean(data, axis=0), config
                )
            }
        else:
            chains = []
            config_labels = config.labels or {}
            for i, d in enumerate(data):
                if labels is not None:
                    config.labels = config_labels.get(labels[i], {})
                if isinstance(d, np.ndarray):
                    chain = DistributionPlotter.prep_from_array(d, config)
                else:
                    chain = DistributionPlotter.prep_from_result(d, config)
                chains.append(chain)
            if labels is None:
                labels = range(len(chains))
            chains = {labels[i]: chain for (i, chain) in enumerate(chains)}
            if (
                "log_posterior" in chain_kwargs
                and chain_kwargs["log_posterior"] is not None
            ):
                log_posterior = chain_kwargs["log_posterior"]
                for name in chains:
                    if isinstance(log_posterior, dict):
                        chains[name]["log_posterior"] = log_posterior.get(name)
                    else:
                        chains[name]["log_posterior"] = log_posterior

        # Clean chains of non-finite rows and zero-variance columns to prevent KDE boundary errors
        cleaned_chains = {}
        for name, chain in chains.items():
            if isinstance(chain, pd.DataFrame):
                # Drop infs/nans
                df = chain.replace([np.inf, -np.inf], np.nan).dropna()
                if len(df) > 1:
                    # Keep only columns that have non-zero variation
                    std = df.std(numeric_only=True)
                    valid_cols = std[std > 1e-8].index
                    if len(valid_cols) > 0:
                        cleaned_chains[name] = df[valid_cols]
            elif isinstance(chain, np.ndarray):
                valid_rows = np.all(np.isfinite(chain), axis=-1)
                arr = chain[valid_rows]
                if len(arr) > 1:
                    ptp = np.ptp(arr, axis=0)
                    valid_cols = np.where(ptp > 1e-8)[0]
                    if len(valid_cols) > 0:
                        cleaned_chains[name] = arr[:, valid_cols]

        if not cleaned_chains:
            cleaned_chains = chains

        try:
            fig, ax = Plotter.corner(cleaned_chains, fig=fig, ax=ax, chain_kwargs=chain_kwargs)
        except Exception as e:
            print("ERROR in Plotter.corner:")
            print(e)
            return None, None
        fig.suptitle((config.plot_kwargs or {}).get("title", config.name.capitalize()))

        if save:
            fig = Plotter.save(fig, savepath)
            Plotter.close(fig, ax)
            return None, None

        return fig, ax
