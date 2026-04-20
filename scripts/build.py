"""Generate static/puzzle_data.json for the frontend.

For each n in a chosen range, we emit:
  - states: list of state strings in V_n order (the longest solution)
  - neighbors: map state -> [(digit_index, neighbor_state), ...]
  - longest_path: list of state strings
  - shortest_path: list of state strings
  - longest_hint: map state -> digit_index of the next V-successor move
  - shortest_hint: map state -> digit_index of the next S-successor move

Also asserts paper-derived counts as a final sanity check.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ziggu.core import (
    enumerate_longest, enumerate_shortest,
    build_state_graph, neighbors, state_to_string,
)


def expected_V(n):
    return (3 ** (n + 1) - 1) // 2

def expected_S(n):
    return 6 * (2 ** n) - 3 * n - 5


def build_for_n(n: int) -> dict:
    longest = enumerate_longest(n)
    shortest = enumerate_shortest(n)

    assert len(longest) == expected_V(n)
    assert len(shortest) == expected_S(n)

    graph = build_state_graph(n)

    # Hints: for each state, the digit index to change next on the path.
    def hint_map(path):
        idx = {s: i for i, s in enumerate(path)}
        h = {}
        for i, s in enumerate(path[:-1]):
            nxt = path[i + 1]
            diffs = [k for k, (a, b) in enumerate(zip(s, nxt)) if a != b]
            assert len(diffs) == 1
            k = diffs[0]
            delta = nxt[k] - s[k]
            h[state_to_string(s)] = {"digit": k, "delta": delta}
        return h

    longest_hint = hint_map(longest)
    shortest_hint = hint_map(shortest)

    return {
        "n": n,
        "states": [state_to_string(s) for s in longest],
        "neighbors": {
            state_to_string(s): [
                {"digit": i, "state": state_to_string(t)} for i, t in nbrs
            ]
            for s, nbrs in graph.items()
        },
        "longest_path": [state_to_string(s) for s in longest],
        "shortest_path": [state_to_string(s) for s in shortest],
        "longest_hint": longest_hint,
        "shortest_hint": shortest_hint,
    }


def main():
    out = {"by_n": {}}
    for n in (2, 3, 4, 5, 6):
        out["by_n"][str(n)] = build_for_n(n)
        print(
            f"n={n}: |V|={expected_V(n)}, |S|={expected_S(n)} "
            f"(states: {len(out['by_n'][str(n)]['states'])})"
        )

    out_path = os.path.join(
        os.path.dirname(__file__), "..", "static", "puzzle_data.json"
    )
    out_path = os.path.abspath(out_path)
    with open(out_path, "w") as f:
        json.dump(out, f, separators=(",", ":"))
    size_kb = os.path.getsize(out_path) / 1024
    print(f"wrote {out_path} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
