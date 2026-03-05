"""Pydantic models for the data processing pipeline."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, ConfigDict


class ValidationResult(BaseModel):
    """Result of processed data validation."""

    model_config = ConfigDict(extra="forbid")

    passed: bool
    warnings: list[str] = []
    stats: dict[str, object] = {}
    timestamp_range: tuple[str, str] = ("", "")


class PipelineResult(BaseModel):
    """Result of a full processing pipeline run."""

    model_config = ConfigDict(extra="forbid")

    interval: str
    input_rows: int
    output_rows: int
    nan_counts: dict[str, int] = {}
    validation: ValidationResult
    output_path: str
    elapsed_seconds: float


class ProcessingConfig(BaseModel):
    """Configuration for the processing pipeline."""

    model_config = ConfigDict(extra="forbid")

    raw_dir: str
    output_dir: str
    intervals: list[str] = ["5m"]
