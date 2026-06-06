from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence
import csv

import numpy as np

from .types import DataSeries, Dataset

_ERROR_COLUMN_NAMES = (
    "y_err",
    "y_error",
    "yerr",
    "sigma",
    "uncertainty",
    "error",
)
_FRACTION_COLUMN_NAMES = (
    "y_err_frac",
    "y_error_frac",
    "frac_err",
    "fractional_error",
    "fraction",
    "rel_err",
    "relative_error",
)


def _coerce_path(path) -> Path:
    return Path(path).expanduser()


def _clean_token(token: str) -> str:
    return token.strip()


def _is_float_token(token: str) -> bool:
    token = token.strip()
    if token == "":
        return False
    try:
        float(token)
        return True
    except ValueError:
        return False


def _read_table(path: Path, delimiter: str | None) -> tuple[list[str], list[list[str]]]:
    rows: list[list[str]] = []
    with path.open(newline="") as handle:
        if delimiter is None:
            for raw_line in handle:
                line = raw_line.split("#", 1)[0].strip()
                if not line:
                    continue
                rows.append(line.split())
        else:
            reader = csv.reader(handle, delimiter=delimiter)
            for raw_row in reader:
                if not raw_row:
                    continue
                joined = ",".join(raw_row) if delimiter == "," else " ".join(raw_row)
                line = joined.split("#", 1)[0].strip()
                if not line:
                    continue
                rows.append([cell.strip() for cell in raw_row])

    if not rows:
        raise ValueError(f"No data found in {path}")

    first_row = rows[0]
    has_header = not all(_is_float_token(token) for token in first_row)
    if has_header:
        header = [_clean_token(token) for token in first_row]
        data_rows = rows[1:]
    else:
        header = [str(i) for i in range(len(first_row))]
        data_rows = rows

    if not data_rows:
        raise ValueError(f"No data rows found in {path}")

    width = len(header)
    normalized_rows: list[list[str]] = []
    for row in data_rows:
        if len(row) != width:
            raise ValueError(f"Inconsistent column count in {path}: expected {width}, got {len(row)}")
        normalized_rows.append(row)
    return header, normalized_rows


def _resolve_column(selector, header: Sequence[str]) -> int:
    if isinstance(selector, int):
        index = selector
    else:
        try:
            index = int(selector)
        except (TypeError, ValueError):
            if selector not in header:
                raise KeyError(f"Column {selector!r} not found") from None
            return header.index(selector)
    if index < 0 or index >= len(header):
        raise IndexError(f"Column index {index} out of range for {len(header)} columns")
    return index


def _extract_column(rows: Sequence[Sequence[str]], index: int) -> np.ndarray:
    return np.asarray([float(row[index]) for row in rows], dtype=float)


def _infer_y_err(y: np.ndarray, *, default_fraction: float, error_mode: str) -> np.ndarray:
    mode = error_mode.lower()
    if mode not in {"auto", "poisson", "fraction"}:
        raise ValueError("error_mode must be one of: auto, poisson, fraction")
    if mode == "poisson":
        return np.sqrt(np.clip(y, 0.0, None))
    if mode == "fraction":
        return np.abs(y) * float(default_fraction)

    finite_y = np.asarray(y, dtype=float)
    if np.all(finite_y >= 0.0) and np.allclose(finite_y, np.round(finite_y)):
        return np.sqrt(np.clip(finite_y, 0.0, None))
    return np.abs(finite_y) * float(default_fraction)


def _load_series(
    path,
    *,
    delimiter: str | None,
    x_col,
    y_col,
    y_err_col=None,
    default_fraction: float = 0.05,
    error_mode: str = "auto",
    label: str = "",
) -> DataSeries:
    path = _coerce_path(path)
    header, rows = _read_table(path, delimiter)

    x_index = _resolve_column(x_col, header)
    y_index = _resolve_column(y_col, header)
    x = _extract_column(rows, x_index)
    y = _extract_column(rows, y_index)

    y_err = None
    sigma_low = None
    sigma_high = None

    sigma_low_names = ("sigma_low", "y_err_low", "y_error_low", "low_err")
    sigma_high_names = ("sigma_high", "y_err_high", "y_error_high", "high_err")
    for candidate in sigma_low_names:
        if candidate in header:
            sigma_low = _extract_column(rows, header.index(candidate))
            break
    for candidate in sigma_high_names:
        if candidate in header:
            sigma_high = _extract_column(rows, header.index(candidate))
            break

    if y_err_col is not None:
        y_err_index = _resolve_column(y_err_col, header)
        y_err = _extract_column(rows, y_err_index)
    else:
        matched = None
        for candidate in _ERROR_COLUMN_NAMES:
            if candidate in header:
                matched = candidate
                break
        if matched is not None:
            y_err = _extract_column(rows, header.index(matched))
        else:
            for candidate in _FRACTION_COLUMN_NAMES:
                if candidate in header:
                    fraction = _extract_column(rows, header.index(candidate))
                    y_err = np.abs(y) * fraction
                    break
            if y_err is None:
                y_err = _infer_y_err(y, default_fraction=default_fraction, error_mode=error_mode)

    return DataSeries(x=x, y=y, y_err=y_err, sigma_low=sigma_low, sigma_high=sigma_high, label=label)


def load_csv(
    path,
    x_col,
    y_col,
    y_err_col=None,
    *,
    default_fraction: float = 0.05,
    error_mode: str = "auto",
    label: str = "",
) -> DataSeries:
    return _load_series(
        path,
        delimiter=",",
        x_col=x_col,
        y_col=y_col,
        y_err_col=y_err_col,
        default_fraction=default_fraction,
        error_mode=error_mode,
        label=label,
    )


def load_txt(
    path,
    x_col,
    y_col,
    y_err_col=None,
    *,
    default_fraction: float = 0.05,
    error_mode: str = "auto",
    label: str = "",
) -> DataSeries:
    return _load_series(
        path,
        delimiter=None,
        x_col=x_col,
        y_col=y_col,
        y_err_col=y_err_col,
        default_fraction=default_fraction,
        error_mode=error_mode,
        label=label,
    )


def combine_series(*series: DataSeries | Dataset | Iterable[DataSeries]) -> Dataset:
    combined: list[DataSeries] = []
    for item in series:
        if isinstance(item, Dataset):
            combined.extend(item.series)
        elif isinstance(item, DataSeries):
            combined.append(item)
        else:
            combined.extend(list(item))
    return Dataset(combined)


__all__ = ["load_csv", "load_txt", "combine_series"]
