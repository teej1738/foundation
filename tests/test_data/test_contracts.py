"""Tests for DataFrame contract validation."""
from __future__ import annotations

import pandas as pd
import pytest

from foundation.data.contracts import (
    ColumnContract,
    ContractViolation,
    DataContract,
    HoldoutViolationError,
    validate_contract,
)


def _make_contract(**overrides) -> DataContract:
    """Create a minimal test contract."""
    defaults = dict(
        name="test_contract",
        version="1.0",
        row_count_range=(1, 100),
        columns=[
            ColumnContract(name="ts", dtype="datetime64[ms, UTC]"),
            ColumnContract(name="price", dtype="float64", min_val=0),
            ColumnContract(name="volume", dtype="float64", min_val=0, nullable=True),
        ],
        timestamp_col="ts",
    )
    defaults.update(overrides)
    return DataContract(**defaults)


def _make_df(n: int = 10) -> pd.DataFrame:
    """Create a minimal valid DataFrame matching the test contract."""
    return pd.DataFrame({
        "ts": pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC"),
        "price": [30000.0 + i for i in range(n)],
        "volume": [100.0 + i for i in range(n)],
    })


class TestValidateContract:
    def test_valid_df_passes(self):
        """A conforming DataFrame passes without error."""
        contract = _make_contract()
        df = _make_df()
        validate_contract(df, contract)  # Should not raise

    def test_missing_column_fails(self):
        """Missing a required column raises ContractViolation."""
        contract = _make_contract()
        df = _make_df().drop(columns=["price"])
        with pytest.raises(ContractViolation, match="Missing column.*price"):
            validate_contract(df, contract)

    def test_row_count_below_range_fails(self):
        """Row count below minimum raises ContractViolation."""
        contract = _make_contract(row_count_range=(100, 200))
        df = _make_df(10)
        with pytest.raises(ContractViolation, match="Row count"):
            validate_contract(df, contract)

    def test_row_count_above_range_fails(self):
        """Row count above maximum raises ContractViolation."""
        contract = _make_contract(row_count_range=(1, 5))
        df = _make_df(10)
        with pytest.raises(ContractViolation, match="Row count"):
            validate_contract(df, contract)

    def test_null_in_non_nullable_fails(self):
        """NaN in a non-nullable column raises ContractViolation."""
        contract = _make_contract()
        df = _make_df()
        df.loc[3, "price"] = float("nan")
        with pytest.raises(ContractViolation, match="NaN"):
            validate_contract(df, contract)

    def test_null_in_nullable_passes(self):
        """NaN in a nullable column does not fail."""
        contract = _make_contract()
        df = _make_df()
        df.loc[3, "volume"] = float("nan")
        validate_contract(df, contract)  # Should not raise

    def test_value_below_min_fails(self):
        """Value below min_val raises ContractViolation."""
        contract = _make_contract()
        df = _make_df()
        df.loc[0, "price"] = -1.0
        with pytest.raises(ContractViolation, match="below min_val"):
            validate_contract(df, contract)

    def test_unsorted_timestamps_fail(self):
        """Non-monotonic timestamps raise ContractViolation."""
        contract = _make_contract()
        df = _make_df()
        # Swap rows to break sort order
        df.iloc[2], df.iloc[3] = df.iloc[3].copy(), df.iloc[2].copy()
        with pytest.raises(ContractViolation, match="not sorted"):
            validate_contract(df, contract)

    def test_sorted_check_skipped_when_disabled(self):
        """must_be_sorted=False skips the sort check."""
        contract = _make_contract(must_be_sorted=False)
        df = _make_df()
        df.iloc[2], df.iloc[3] = df.iloc[3].copy(), df.iloc[2].copy()
        validate_contract(df, contract)  # Should not raise


class TestHoldoutViolationError:
    def test_is_importable_and_raisable(self):
        """HoldoutViolationError can be raised and caught."""
        with pytest.raises(HoldoutViolationError, match="test"):
            raise HoldoutViolationError("test holdout violation")
