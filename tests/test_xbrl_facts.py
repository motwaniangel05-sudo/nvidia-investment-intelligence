"""Tests for XBRL company-facts flattening and filtering."""

import pandas as pd

from data_acquisition.xbrl_facts import filter_by_year, flatten_facts


def fake_facts_json():
    return {
        "entityName": "NVIDIA CORP",
        "facts": {
            "us-gaap": {
                "Revenues": {
                    "units": {
                        "USD": [
                            {"start": "2014-02-01", "end": "2015-01-31",
                             "val": 4682000000, "form": "10-K",
                             "fy": 2015, "fp": "FY", "filed": "2015-03-01",
                             "accn": "0001-15-000001"},
                            {"start": "2015-02-01", "end": "2016-01-31",
                             "val": 5010000000, "form": "10-K",
                             "fy": 2016, "fp": "FY", "filed": "2016-03-17",
                             "accn": "0001-16-000205"},
                            {"start": "2016-02-01", "end": "2016-04-30",
                             "val": 1303000000, "form": "10-Q",
                             "fy": 2016, "fp": "Q1", "filed": "2016-05-20",
                             "accn": "0001-16-000300"},
                        ]
                    }
                },
                "NetIncomeLoss": {
                    "units": {
                        "USD": [
                            {"start": "2015-02-01", "end": "2016-01-31",
                             "val": 614000000, "form": "10-K",
                             "fy": 2016, "fp": "FY", "filed": "2016-03-17",
                             "accn": "0001-16-000205"},
                        ]
                    }
                },
                # A concept NOT in our allowlist, to test filtering
                "ShareBasedCompensation": {
                    "units": {
                        "USD": [
                            {"start": "2015-02-01", "end": "2016-01-31",
                             "val": 200000000, "form": "10-K",
                             "fy": 2016, "fp": "FY", "filed": "2016-03-17",
                             "accn": "0001-16-000205"},
                        ]
                    }
                },
            }
        },
    }


def test_flatten_produces_expected_row_count():
    df = flatten_facts(fake_facts_json())
    assert len(df) == 5  # 3 Revenues + 1 NetIncomeLoss + 1 ShareBasedCompensation


def test_flatten_has_expected_columns():
    df = flatten_facts(fake_facts_json())
    expected_cols = {
        "concept", "unit", "value", "start", "end",
        "form", "fy", "fp", "filed", "accn", "frame",
    }
    assert expected_cols.issubset(set(df.columns))


def test_flatten_with_concept_allowlist_filters_out_others():
    df = flatten_facts(fake_facts_json(), concepts=["Revenues", "NetIncomeLoss"])
    assert set(df["concept"].unique()) == {"Revenues", "NetIncomeLoss"}
    assert "ShareBasedCompensation" not in df["concept"].values


def test_flatten_empty_json_returns_empty_df_with_columns():
    df = flatten_facts({})
    assert df.empty
    assert "concept" in df.columns


def test_flatten_missing_concept_in_allowlist_is_just_absent():
    # Ask for a concept that doesn't exist in the fake data at all
    df = flatten_facts(fake_facts_json(), concepts=["ThisConceptDoesNotExist"])
    assert df.empty


def test_values_are_correct_for_known_row():
    df = flatten_facts(fake_facts_json(), concepts=["NetIncomeLoss"])
    row = df.iloc[0]
    assert row["value"] == 614000000
    assert row["form"] == "10-K"
    assert row["fy"] == 2016


def test_filter_by_year_keeps_only_requested_range():
    df = flatten_facts(fake_facts_json(), concepts=["Revenues"])
    out = filter_by_year(df, 2016, 2016)
    # Only rows whose 'end' falls in 2016: the FY2016 10-K and the Q1 10-Q
    assert len(out) == 2
    assert (out["end"].dt.year == 2016).all()


def test_filter_by_year_excludes_out_of_range():
    df = flatten_facts(fake_facts_json(), concepts=["Revenues"])
    out = filter_by_year(df, 2020, 2026)
    assert out.empty


def test_filter_by_year_on_empty_df_returns_empty():
    empty = pd.DataFrame(columns=["end"])
    out = filter_by_year(empty, 2016, 2020)
    assert out.empty
