from pathlib import Path

import numpy as np

from labfit.io import combine_series, load_csv, load_txt
from labfit.types import DataSeries, Dataset


def test_csv_and_txt_loaders_infer_errors_and_stack_series(tmp_path: Path):
    csv = tmp_path / "sample.csv"
    csv.write_text(
        "time,count,sigma_low,sigma_high\n"
        "0,1,0.1,0.2\n"
        "1,4,0.2,0.3\n"
    )
    csv_series = load_csv(csv, "time", "count")
    assert isinstance(csv_series, DataSeries)
    assert np.allclose(csv_series.x, [0.0, 1.0])
    assert np.allclose(csv_series.y, [1.0, 4.0])
    assert np.allclose(csv_series.sigma_low, [0.1, 0.2])
    assert np.allclose(csv_series.sigma_high, [0.2, 0.3])
    assert np.allclose(csv_series.y_err, [1.0, 2.0])

    txt = tmp_path / "sample.txt"
    txt.write_text(
        "x intensity\n"
        "0 10\n"
        "1 12.5\n"
    )
    txt_series = load_txt(txt, "x", "intensity")
    assert np.allclose(txt_series.x, [0.0, 1.0])
    assert np.allclose(txt_series.y, [10.0, 12.5])
    assert np.allclose(txt_series.y_error, [0.5, 0.625])

    dataset = combine_series(csv_series, txt_series)
    assert isinstance(dataset, Dataset)
    assert len(dataset) == 2
    assert np.allclose(dataset[0].x, csv_series.x)
    assert np.allclose(dataset[1].y, txt_series.y)
