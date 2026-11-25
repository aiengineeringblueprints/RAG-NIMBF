"""Shared validation of the JSON-object lineage knobs (issue #65)."""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence


def parse_json_object(
    value: str | Mapping[str, Any] | None,
    *,
    setting: str,
    label: str,
    allowed: Sequence[str],
) -> dict[str, Any] | None:
    """Parse ``value`` into a dict whose keys are all in ``allowed``.

    ``None`` when unset. ``setting`` names the config field in errors,
    ``label`` names the kind of key ("experiment metadata").
    """
    if value is None or value == "":
        return None
    data = json.loads(value) if isinstance(value, str) else dict(value)
    if not isinstance(data, dict):
        raise ValueError(f"{setting} must contain a JSON object")
    unknown = sorted(set(data) - set(allowed))
    if unknown:
        raise ValueError(
            f"Unknown {label} field(s): {', '.join(unknown)}; allowed: {', '.join(allowed)}"
        )
    return data
