"""MTO-Front-Identifier: Objective meteorological front identification and plotting."""

from src.identifier import (
    IDENTIFICATION_METHODS,
    classify_front_type,
    identify_fronts,
    identify_fronts_by_method,
    identify_fronts_f_diagnostic,
    identify_fronts_frontogenesis,
    identify_fronts_tfp,
)
from src.plot import (
    plot_classified_fronts,
    plot_front_samples,
    plot_method_comparison,
)
from src.utils import (
    load_era5_datasets,
    preprocess_era5_variable,
    preprocess_era5_variables,
)

__all__ = [
    "IDENTIFICATION_METHODS",
    "classify_front_type",
    "identify_fronts",
    "identify_fronts_by_method",
    "identify_fronts_f_diagnostic",
    "identify_fronts_frontogenesis",
    "identify_fronts_tfp",
    "load_era5_datasets",
    "plot_classified_fronts",
    "plot_front_samples",
    "plot_method_comparison",
    "preprocess_era5_variable",
    "preprocess_era5_variables",
]
