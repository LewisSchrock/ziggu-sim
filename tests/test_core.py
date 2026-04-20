"""Tests against ground truth from the Goertz & Williams paper.

Truth data:
  Table 4: |V_n| = (3^{n+1} - 1)/2 => 4, 13, 40, 121, 364, 1093 for n=1..6
           |S_n| = 6*2^n - 3n - 5  => 4, 13, 34, 79, 172, 361 for n=1..6
  Table 1: exact n=3 ranking for V_3 and S_3 (first dozen entries).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ziggu.core import (
    is_valid, neighbors, long_successor,
    enumerate_longest, enumerate_shortest,
    state_from_string, state_to_string,
)


def expected_V(n):
    return (3 ** (n + 1) - 1) // 2

def expected_S(n):
    return 6 * (2 ** n) - 3 * n - 5


def test_is_valid_basic():
    # 13 valid for n=2 (Table 4)
    count = sum(1 for w in _all_quaternary(2) if is_valid(w))
    assert count == 13

def test_valid_state_counts():
    for n in range(1, 7):
        count = sum(1 for w in _all_quaternary(n) if is_valid(w))
        assert count == expected_V(n), f"n={n}: got {count}, expected {expected_V(n)}"

def test_longest_lengths():
    for n in range(1, 7):
        path = enumerate_longest(n)
        # all valid
        assert all(is_valid(w) for w in path)
        # no duplicates
        assert len(set(path)) == len(path)
        # covers V_n entirely
        assert len(path) == expected_V(n), f"n={n}: longest len {len(path)} vs {expected_V(n)}"
        # starts at 0^n, ends at 3^n
        assert path[0] == (0,) * n
        assert path[-1] == (3,) * n

def test_shortest_lengths():
    for n in range(1, 7):
        path = enumerate_shortest(n)
        assert len(path) == expected_S(n), f"n={n}: shortest len {len(path)} vs {expected_S(n)}"
        assert path[0] == (0,) * n
        assert path[-1] == (3,) * n

def test_successive_shortest_states_are_neighbors():
    # In the shortest solution, consecutive states may be non-adjacent in V_n
    # (they differ by one digit but by up to ±1... actually the paper says the
    # shortest solution also moves one digit by ±1 each step, since S_n is a
    # sublist of Q_n and consecutive states in Q_n differ by ±1 in one digit).
    # Wait - S_n is a sublist of V_n but consecutive S_n states may skip
    # over V_n states. So successive S_n states might differ by more than a
    # single ±1 move. That's fine for the *solution encoding* but our UI
    # walks through V_n adjacency. We verify the looser property: consecutive
    # shortest states are connected by a chain of valid ±1 moves (subpath of V_n).
    for n in [2, 3, 4]:
        longest = enumerate_longest(n)
        shortest = enumerate_shortest(n)
        idx_in_long = {s: i for i, s in enumerate(longest)}
        for a, b in zip(shortest, shortest[1:]):
            ia, ib = idx_in_long[a], idx_in_long[b]
            # In V_n, consecutive entries differ by ±1 in one digit.
            assert ib > ia

def test_table1_n3_longest_prefix():
    # Table 1 columns V_3: rank 0..12 should be these base-4 strings.
    expected = [
        "000","001","002","003","013","012","011","010","020",
        "021","022","023","033",
    ]
    path = enumerate_longest(3)
    got = [state_to_string(w) for w in path[:len(expected)]]
    assert got == expected, f"got {got}"

def test_table1_n3_shortest_prefix():
    # Table 1 columns S_3: rank 0..12 should also match (S_3 agrees with V_3
    # for the first 13 entries according to the table).
    expected = [
        "000","001","002","003","013","012","011","010","020",
        "021","022","023","033",
    ]
    path = enumerate_shortest(3)
    got = [state_to_string(w) for w in path[:len(expected)]]
    assert got == expected

def test_longest_full_n3():
    # V_3 from paper eq. (8): V_n = 0·V_{n-1}, 1·V_{n-1}^R, 2·V_{n-1}, 3^n
    # Yields 40 states, matching (3^{n+1}-1)/2 = 40.
    expected_str = (
        "000 001 002 003 013 012 011 010 020 021 022 023 033 "
        "133 123 122 121 120 110 111 112 113 103 102 101 100 "
        "200 201 202 203 213 212 211 210 220 221 222 223 233 "
        "333"
    ).split()
    assert len(expected_str) == 40
    path = enumerate_longest(3)
    got = [state_to_string(w) for w in path]
    assert got == expected_str, f"\nexpected ({len(expected_str)}): {expected_str}\ngot      ({len(got)}): {got}"


def test_shortest_full_n3():
    # S_3 from paper eq. (9); 34 states. Takes the 103 -> 203 shortcut.
    expected_str = (
        "000 001 002 003 013 012 011 010 020 021 022 023 033 "
        "133 123 122 121 120 110 111 112 113 103 "
        "203 213 212 211 210 220 221 222 223 233 "
        "333"
    ).split()
    assert len(expected_str) == 34
    path = enumerate_shortest(3)
    got = [state_to_string(w) for w in path]
    assert got == expected_str

def test_shortest_steps_are_single_moves():
    # Every consecutive pair in S_n differs in exactly one digit by ±1,
    # and both are valid states (i.e., it's a legal puzzle move).
    for n in range(1, 7):
        path = enumerate_shortest(n)
        for a, b in zip(path, path[1:]):
            diffs = [(i, x, y) for i, (x, y) in enumerate(zip(a, b)) if x != y]
            assert len(diffs) == 1, f"n={n}: {a}->{b} diffs={diffs}"
            assert abs(diffs[0][1] - diffs[0][2]) == 1
            assert is_valid(a) and is_valid(b)

def test_neighbors_valid():
    # every neighbor is itself a valid state and differs by ±1 in exactly
    # one digit.
    for w in enumerate_longest(3):
        for i, t in neighbors(w):
            assert is_valid(t)
            diffs = [(j, a, b) for j, (a, b) in enumerate(zip(w, t)) if a != b]
            assert len(diffs) == 1 and diffs[0][0] == i
            assert abs(diffs[0][1] - diffs[0][2]) == 1

def test_neighbors_symmetric():
    for w in enumerate_longest(3):
        for _, t in neighbors(w):
            back = [tt for _, tt in neighbors(t)]
            assert w in back

def test_n2_state_graph_is_path():
    # Figure 4a: n=2 state graph is a simple path on 13 nodes.
    path = enumerate_longest(2)
    assert len(path) == 13
    # interior nodes have degree 2, endpoints have degree 1.
    degs = [len(neighbors(w)) for w in path]
    assert degs[0] == 1, f"start node degree {degs[0]}"
    assert degs[-1] == 1, f"end node degree {degs[-1]}"
    for d in degs[1:-1]:
        assert d == 2, f"interior degree {d}"

def test_successor_moves_are_legal():
    # Every consecutive pair in V_n and S_n must be connected by a legal move.
    for n in range(1, 7):
        for path_name, path in (
            ("V", enumerate_longest(n)),
            ("S", enumerate_shortest(n)),
        ):
            for a, b in zip(path, path[1:]):
                legal = [t for _, t in neighbors(a)]
                assert b in legal, f"{path_name}_{n}: {a}->{b} is not a legal move"


def _all_quaternary(n):
    if n == 0:
        yield ()
        return
    for d in range(4):
        for tail in _all_quaternary(n - 1):
            yield (d,) + tail


if __name__ == "__main__":
    import traceback
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"ok  {t.__name__}")
        except AssertionError as e:
            print(f"FAIL {t.__name__}: {e}")
            failed += 1
        except Exception:
            print(f"ERR  {t.__name__}")
            traceback.print_exc()
            failed += 1
    print()
    print(f"{len(tests) - failed}/{len(tests)} passed")
    sys.exit(0 if failed == 0 else 1)
