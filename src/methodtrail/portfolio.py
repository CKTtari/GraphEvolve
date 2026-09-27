"""Keeps non-dominated experiment variants instead of only a single best result."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field


class VariantProfile(BaseModel):
    variant_id: str
    metric: float
    wall_seconds: float = Field(ge=0.0)
    reliability: float = Field(ge=0.0, le=1.0)
    information_gain: float = Field(ge=0.0)
    failure_risk: float = Field(ge=0.0, le=1.0)
    signature: str


class PortfolioManager:
    """Persist a compact Pareto set over quality, cost, reliability, and information value."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.profiles: dict[str, VariantProfile] = {}
        if self.path.exists():
            self.profiles = {
                item["variant_id"]: VariantProfile.model_validate(item)
                for item in json.loads(self.path.read_text(encoding="utf-8"))
            }

    def register(self, profile: VariantProfile) -> None:
        self.profiles[profile.variant_id] = profile
        self._save()

    def pareto(self, limit: int = 12, maximize_metric: bool = True) -> list[VariantProfile]:
        profiles = list(self.profiles.values())
        kept = [
            profile
            for profile in profiles
            if not any(
                self._dominates(other, profile, maximize_metric)
                for other in profiles
                if other.variant_id != profile.variant_id
            )
        ]
        # Keep diverse signatures when several points have equivalent tradeoffs.
        kept.sort(
            key=lambda item: (
                item.metric if maximize_metric else -item.metric,
                item.reliability,
                item.information_gain,
                -item.wall_seconds,
            ),
            reverse=True,
        )
        seen_signatures: set[str] = set()
        diverse = []
        for profile in kept:
            if profile.signature not in seen_signatures or len(diverse) < 3:
                diverse.append(profile)
                seen_signatures.add(profile.signature)
            if len(diverse) >= limit:
                break
        return diverse

    @staticmethod
    def _dominates(
        left: VariantProfile, right: VariantProfile, maximize_metric: bool = True
    ) -> bool:
        metric_no_worse = (
            left.metric >= right.metric
            if maximize_metric
            else left.metric <= right.metric
        )
        metric_strictly_better = (
            left.metric > right.metric
            if maximize_metric
            else left.metric < right.metric
        )
        no_worse = (
            metric_no_worse
            and left.reliability >= right.reliability
            and left.information_gain >= right.information_gain
            and left.wall_seconds <= right.wall_seconds
            and left.failure_risk <= right.failure_risk
        )
        strictly_better = (
            metric_strictly_better
            or left.reliability > right.reliability
            or left.information_gain > right.information_gain
            or left.wall_seconds < right.wall_seconds
            or left.failure_risk < right.failure_risk
        )
        return no_worse and strictly_better

    def _save(self) -> None:
        payload = [
            profile.model_dump(mode="json") for profile in self.profiles.values()
        ]
        self.path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
