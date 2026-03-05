"""Tests for embargo validation (AD-28)."""
from __future__ import annotations

import datetime as dt

import numpy as np
import pytest

from foundation.data.embargo import validate_embargo, validate_no_index_overlap


class TestValidateEmbargo:

    def test_valid_embargo(self):
        train_end = dt.datetime(2020, 6, 1)
        test_start = dt.datetime(2020, 6, 4)
        result = validate_embargo(train_end, test_start, embargo_days=2)
        assert result.valid
        assert len(result.violations) == 0

    def test_exact_embargo_boundary(self):
        train_end = dt.datetime(2020, 6, 1)
        test_start = dt.datetime(2020, 6, 3)
        result = validate_embargo(train_end, test_start, embargo_days=2)
        assert result.valid

    def test_violated_embargo(self):
        train_end = dt.datetime(2020, 6, 1)
        test_start = dt.datetime(2020, 6, 2)
        result = validate_embargo(train_end, test_start, embargo_days=3)
        assert not result.valid
        assert len(result.violations) == 1
        assert "Embargo violated" in result.violations[0]

    def test_zero_embargo(self):
        train_end = dt.datetime(2020, 6, 1)
        test_start = dt.datetime(2020, 6, 1)
        result = validate_embargo(train_end, test_start, embargo_days=0)
        assert result.valid

    def test_large_embargo_with_small_gap(self):
        train_end = dt.datetime(2020, 6, 1)
        test_start = dt.datetime(2020, 6, 5)
        result = validate_embargo(train_end, test_start, embargo_days=30)
        assert not result.valid


class TestValidateNoIndexOverlap:

    def test_no_overlap(self):
        train = np.array([0, 1, 2, 3])
        test = np.array([5, 6, 7, 8])
        result = validate_no_index_overlap(train, test)
        assert result.valid
        assert len(result.violations) == 0

    def test_overlap_detected(self):
        train = np.array([0, 1, 2, 3, 4])
        test = np.array([3, 4, 5, 6])
        result = validate_no_index_overlap(train, test)
        assert not result.valid
        assert "overlap" in result.violations[0].lower()

    def test_empty_arrays(self):
        train = np.array([], dtype=int)
        test = np.array([], dtype=int)
        result = validate_no_index_overlap(train, test)
        assert result.valid
