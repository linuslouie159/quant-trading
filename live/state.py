"""Tiny JSON state store for the live alerting layer -- READ/WRITE local files only.

The Telegram bot itself is stateless (it pulls live state from Alpaca on demand).
But two ALERTS need to remember the previous run to know if anything changed:

  * signal-change alert  -> needs yesterday's signalled-long set per strategy
  * drawdown alert       -> needs the running peak equity per account

This module is a minimal key->JSON-file store under live/state/ (gitignored). It
holds NO secrets and NO trading logic -- just small dictionaries. Reads fail soft
(return the supplied default) so a missing/corrupt file never breaks a daily run.

    from live.state import load, save
    last = load("last_signal", {})            # {} if not written yet
    save("last_signal", {"trend_momentum": ["SPY", "QQQ"]})
"""

from __future__ import annotations

import json
import os

_STATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "state")


def _path(name: str) -> str:
    return os.path.join(_STATE_DIR, f"{name}.json")


def load(name: str, default):
    """Return the parsed JSON for `name`, or `default` if missing/unreadable."""
    try:
        with open(_path(name), "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return default


def save(name: str, obj) -> None:
    """Write `obj` as JSON to live/state/<name>.json, creating the dir if needed."""
    os.makedirs(_STATE_DIR, exist_ok=True)
    # Write to a temp file then replace, so a crash mid-write can't corrupt state.
    tmp = _path(name) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
    os.replace(tmp, _path(name))
