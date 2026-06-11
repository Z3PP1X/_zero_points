import pandas as pd
import pytest

from supervised_learning.dataset import DatasetLoader, canonical_dataset_column


@pytest.mark.parametrize(
    ("raw_header", "canonical"),
    [
        ("x0", "x0"),
        ("X0", "x0"),
        ("f(x0)", "fx"),
        ("F(x0)", "fx"),
        ("f'(x0)", "d1x"),
        ("f''(x0)", "d2x"),
        ("f(x_0)", "fx"),
        ("f'(x_0)", "d1x"),
        ("f''(x_0)", "d2x"),
        ("fx", "fx"),
        ("d1x", "d1x"),
        ("d2x", "d2x"),
        ("dfx", "d1x"),
        ("d2fx", "d2x"),
        ("startwert", "x0"),
        ("problem_id", "problem_id"),
    ],
)
def test_canonical_dataset_column_maps_headers(raw_header, canonical):
    assert canonical_dataset_column(raw_header) == canonical


def test_canonical_dataset_column_does_not_collapse_derivative_headers():
    assert canonical_dataset_column("f(x0)") == "fx"
    assert canonical_dataset_column("f'(x0)") == "d1x"
    assert canonical_dataset_column("f''(x0)") == "d2x"


def test_dataset_loader_normalizes_mathematica_headers(tmp_path):
    csv_path = tmp_path / "sample.csv"
    csv_path.write_text(
        "problem_id,point_index,x0,f(x0),f'(x0),f''(x0),y_target,"
        "Newton_absTime,GMGF_absTime,Newton_iterSteps,GMGF_iterSteps\n"
        "P1,0,1.5,10.0,2.0,-0.5,12.0,1.0,2.0,3,4\n",
        encoding="utf-8",
    )

    loader = DatasetLoader(
        dataset_name="sample",
        run_key="ignored",
        base_dir=tmp_path,
    )
    df = loader.data

    assert "fx" in df.columns
    assert "d1x" in df.columns
    assert "d2x" in df.columns
    assert "f(x0)" not in df.columns
    assert "f'(x0)" not in df.columns
    assert "f''(x0)" not in df.columns
    assert df.loc[0, "x0"] == pytest.approx(1.5)
    assert df.loc[0, "fx"] == pytest.approx(10.0)
    assert df.loc[0, "d1x"] == pytest.approx(2.0)
    assert df.loc[0, "d2x"] == pytest.approx(-0.5)

