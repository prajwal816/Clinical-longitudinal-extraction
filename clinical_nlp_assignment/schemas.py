from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


StatusKey = Literal["active", "resolved", "suspected"]


class Evidence(BaseModel):
    model_config = ConfigDict(extra="ignore")

    note_id: str
    line_no: int = Field(ge=1)
    span: str


class Condition(BaseModel):
    model_config = ConfigDict(extra="ignore")

    condition_name: str = Field(min_length=1)
    category: str = Field(min_length=1)
    subcategory: str = Field(min_length=1)
    status: StatusKey
    onset: str | None = None
    evidence: list[Evidence] = Field(default_factory=list)

    @field_validator("evidence")
    @classmethod
    def _non_empty_evidence(cls, v: list[Evidence]) -> list[Evidence]:
        if not v:
            raise ValueError("evidence must be non-empty")
        return v


class PatientOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    patient_id: str
    conditions: list[Condition] = Field(default_factory=list)


class Taxonomy(BaseModel):
    """
    Minimal runtime view of taxonomy.json, used for validation.
    """

    model_config = ConfigDict(extra="allow")

    condition_categories: dict[str, Any]
    status_values: dict[str, Any]

    @model_validator(mode="after")
    def _validate_shape(self) -> "Taxonomy":
        if not isinstance(self.condition_categories, dict) or not self.condition_categories:
            raise ValueError("taxonomy.condition_categories must be a non-empty dict")
        if not isinstance(self.status_values, dict) or not self.status_values:
            raise ValueError("taxonomy.status_values must be a non-empty dict")
        return self


def validate_condition_taxonomy(
    condition: Condition, valid_category_to_subcats: dict[str, list[str]]
) -> None:
    if condition.category not in valid_category_to_subcats:
        raise ValueError(f"Invalid category: {condition.category}")
    if condition.subcategory not in valid_category_to_subcats[condition.category]:
        raise ValueError(
            f"Invalid subcategory for {condition.category}: {condition.subcategory}"
        )

