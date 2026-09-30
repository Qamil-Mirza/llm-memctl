"""Small helpers with no dependencies inside memctl."""

from __future__ import annotations


def merged(defaults: dict, overrides: dict) -> dict:
    """`defaults` with `overrides` on top; nested dictionaries are merged, not replaced."""
    result = dict(defaults)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merged(result[key], value)
        else:
            result[key] = value
    return result


def set_dotted(config: dict, dotted_key: str, value) -> None:
    """Set `config["a"]["b"]["c"]` from the key "a.b.c", creating dictionaries on the way."""
    *parents, last = dotted_key.split(".")
    node = config
    for key in parents:
        node = node.setdefault(key, {})
    node[last] = value
