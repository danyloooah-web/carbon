"""Shared helpers for the dspd ops: one-time liveness lines and source pins of the engine functions an op mirrors."""
import hashlib
import inspect
import os
import sys

_said = set()


def say(tag, what):
    """One line per (tag, message) per process on stderr: which path a lever took, for the A/B record."""
    key = (tag, what)
    if key not in _said:
        _said.add(key)
        print(f"[dspd {tag}] pid {os.getpid()}: {what}", file=sys.stderr, flush=True)


def digest(fn):
    """First 16 hex of the sha256 of a function's source, or '' when the source is not readable."""
    try:
        return hashlib.sha256(inspect.getsource(fn).encode()).hexdigest()[:16]
    except (OSError, TypeError):
        return ""


def pins_match(tag, expected, functions):
    """True when every mirrored engine function still has a source the op was written against (expected maps a name
    to one digest or to a set of accepted digests: the engine versions the op's body was checked against)."""
    got = {name: digest(fn) for name, fn in functions.items()}
    ok = set(got) == set(expected) and all(
        got[n] in (expected[n] if isinstance(expected[n], (set, frozenset)) else {expected[n]}) for n in got)
    say(tag, "engine pins match" if ok else f"engine pins MISMATCH {got}: lever off, stock forward")
    return ok
