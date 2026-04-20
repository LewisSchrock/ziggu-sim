"""Core logic for Ziggu puzzles.

State encoding: base-4 string w = c1 r1 r2 ... rm of length n = m+1.
By convention we store states as tuples of ints, highest index on the left:
    w = (w[n-1], w[n-2], ..., w[1], w[0])  == q_n q_{n-1} ... q_1 in the paper.

A Ziggu-valid state (member of V_n) is one where no '3' digit is followed
(to its right, i.e. in lower indices) by a non-'3' digit. Geometrically
this enforces the forbidden cells (3, 0..2) in each maze, since a '3'
in the row position locks the column to the maze exit axis.

See Goertz & Williams, "The Quaternary Gray Code and Ziggu Puzzles"
(FUN 2026), Section 4 and eq. (11).
"""
from __future__ import annotations

from typing import Iterator, List, Tuple, Dict, Optional

State = Tuple[int, ...]  # q_n, q_{n-1}, ..., q_1  (high index first)


# ---------------------------------------------------------------------------
# Validity
# ---------------------------------------------------------------------------

def is_valid(w: State) -> bool:
    """Return True iff w is a valid Ziggu state (w in V_n).

    Valid iff for every position where the digit is 3, every digit to its
    right in the paper writing (= every later index in our tuple) is also
    3. Equivalently, once we see a 3 scanning left-to-right, every
    subsequent digit must also be 3.

    Geometrically: in each maze i, the cell (r_i, c_i) is valid iff
    r_i != 3 or c_i = 3. Since our tuple stores c_1 r_1 r_2 ... r_m
    high-index-first (so r_m, r_{m-1}, ..., r_1, c_1), adjacent tuple
    entries are the (r, c) pair for the same maze, and "r=3 -> c=3"
    propagated transitively gives this condition.
    """
    seen_three = False
    for d in w:
        if seen_three:
            if d != 3:
                return False
        elif d == 3:
            seen_three = True
    return True


# ---------------------------------------------------------------------------
# Neighbors (physically reachable states by a single ±1 digit change)
# ---------------------------------------------------------------------------

def _row_change_legal(r_from: int, r_to: int, c: int) -> bool:
    """Row transition in the S-maze: 0<->1 at c=3; 1<->2 at c=0; 2<->3 at c=3."""
    if abs(r_from - r_to) != 1:
        return False
    pair = frozenset((r_from, r_to))
    if pair == frozenset({0, 1}): return c == 3
    if pair == frozenset({1, 2}): return c == 0
    if pair == frozenset({2, 3}): return c == 3
    return False


def _col_change_legal(c_from: int, c_to: int, r: int) -> bool:
    """Column transition: any c<->c±1 step is legal in row 0,1,2; none in row 3."""
    if abs(c_from - c_to) != 1:
        return False
    return r in (0, 1, 2)


def neighbors(w: State) -> List[Tuple[int, State]]:
    """Legal single-move neighbors of w.

    A move changes one tuple index w[k] by ±1. That digit corresponds to
    either c_1 (k = n-1), r_m (k = 0), or simultaneously r_i and c_{i+1}
    (0 < k < n-1, with i = n-1-k). A move is legal iff:
      (a) the resulting state is valid, and
      (b) the induced S-path step in EACH affected maze is legal (using
          the row/column transition rules above).

    Returns a list of (digit_index, new_state).
    """
    out = []
    n = len(w)
    for k in range(n):
        for delta in (-1, +1):
            new_d = w[k] + delta
            if not (0 <= new_d <= 3):
                continue
            new_state = w[:k] + (new_d,) + w[k + 1:]
            if not is_valid(new_state):
                continue

            # n=1 is degenerate (no mazes); every ±1 is legal.
            if n == 1:
                out.append((k, new_state))
                continue

            if k == n - 1:
                # Changing c_1 only; maze 1 column change, r_1 = w[n-2]
                if n < 2 or not _col_change_legal(w[k], new_d, w[n - 2]):
                    continue
            elif k == 0:
                # Changing r_m only; maze m row change, c_m = w[1]
                if n < 2 or not _row_change_legal(w[k], new_d, w[1]):
                    continue
            else:
                # Changing r_i (maze i row) AND c_{i+1} (maze i+1 column)
                # c_i = w[k+1] (for maze i row change)
                # r_{i+1} = w[k-1] (for maze i+1 column change)
                if not _row_change_legal(w[k], new_d, w[k + 1]):
                    continue
                if not _col_change_legal(w[k], new_d, w[k - 1]):
                    continue

            out.append((k, new_state))
    return out


# ---------------------------------------------------------------------------
# Longest-solution successor (quaternary reflected Gray code restricted to V_n)
# ---------------------------------------------------------------------------
#
# Paper eq. (11): let q = q_n ... q_1. The long successor increments or
# decrements digit q_i where i is the smallest index such that:
#   - sum(q_{i+1}..q_n) is even  and q_i < 3                        -> increment
#   - sum(q_{i+1}..q_n) is odd   and q_i > 0 and q_{i-1} != 3       -> decrement
#
# If no such i exists, the successor is undefined (we've reached 3^n).
#
# In our storage w[0] = q_n (highest), w[-1] = q_1 (lowest).
# So "digit q_i" corresponds to w[n - i]; "digits to the right" (higher paper
# index -> smaller paper index? careful) - actually in the paper, q_n is on
# the LEFT (written first) and q_1 on the right. "sum of q_{i+1}..q_n" means
# sum of the digits to the LEFT of q_i in the written form, i.e. earlier
# entries in our tuple.
#
# Let's redo with our tuple indexing directly:
#   - paper i ranges 1..n, paper writes q_n q_{n-1} ... q_1
#   - our tuple w has w[k] = q_{n-k}, k in 0..n-1
#   - paper "sum j>i q_j" = sum of w[k] for k such that n-k > i, i.e. k < n-i
#     = sum of w[0:n-i]   (entries strictly to the LEFT of position of q_i)
#   - paper "q_{i-1}" = w[n - (i-1)] = w[n - i + 1]  (the next entry to the
#     RIGHT in our tuple)
#
# We pick the smallest paper i satisfying the condition, which is the
# RIGHTMOST position in our tuple. So we iterate our tuple from right to
# left (k = n-1 down to 0) and take the first match.
# ---------------------------------------------------------------------------

def long_successor(w: State) -> Optional[State]:
    """Return the successor of w in the longest solution V_n, or None.

    Picks the smallest paper index i (rightmost position in our tuple) such
    that the parity-dictated ±1 move on that digit lands in a valid state.
    Parity: let s = sum of digits strictly to the LEFT of position i in the
    paper writing (= digits to the LEFT in our tuple). If s is even, we try
    to increment; if odd, decrement. A move is allowed iff the resulting
    digit is in [0,3] and the resulting state is in V_n.

    This is equivalent to paper eq. (11); we check validity directly on the
    candidate state rather than encoding the equivalent side-conditions.
    """
    n = len(w)
    prefix = [0] * (n + 1)
    for k in range(n):
        prefix[k + 1] = prefix[k] + w[k]
    for k in range(n - 1, -1, -1):
        s_left = prefix[k]
        q_i = w[k]
        if s_left % 2 == 0:
            if q_i < 3:
                candidate = w[:k] + (q_i + 1,) + w[k + 1:]
                if is_valid(candidate):
                    return candidate
        else:
            if q_i > 0:
                candidate = w[:k] + (q_i - 1,) + w[k + 1:]
                if is_valid(candidate):
                    return candidate
    return None


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

def enumerate_longest(n: int) -> List[State]:
    """Return the longest solution V_n as a list starting at (0,)*n."""
    start: State = (0,) * n
    path = [start]
    while True:
        nxt = long_successor(path[-1])
        if nxt is None:
            break
        path.append(nxt)
    return path


def _core(n: int) -> List[State]:
    """Recursive helper from paper eq. (9):
    core(1) = 0,1,2,3
    core(n) = 03^{n-1}, 1 · core(n-1)^R, 2 · core(n-1), 3^n
    """
    if n == 1:
        return [(0,), (1,), (2,), (3,)]
    prev = _core(n - 1)
    result: List[State] = [(0,) + (3,) * (n - 1)]
    result.extend((1,) + s for s in reversed(prev))
    result.extend((2,) + s for s in prev)
    result.append((3,) * n)
    return result


def enumerate_shortest(n: int) -> List[State]:
    """Return the shortest solution S_n per paper eq. (9):
    S_1 = 0,1,2,3
    S_n = 0 · S_{n-1}, core(n)[1:]

    Note: S_n is NOT obtained by filtering V_n; it takes valid ±1 shortcuts
    that skip certain V_n states (e.g., for n=3, S_3 jumps directly from
    103 to 203, skipping the detour 102,101,100,200,201,202).
    """
    if n == 1:
        return [(0,), (1,), (2,), (3,)]
    prev = enumerate_shortest(n - 1)
    result: List[State] = [(0,) + s for s in prev]
    result.extend(_core(n)[1:])
    return result


# ---------------------------------------------------------------------------
# State graph (for the graph view)
# ---------------------------------------------------------------------------

def build_state_graph(n: int) -> Dict[State, List[Tuple[int, State]]]:
    """Return {state: [(digit_idx, neighbor), ...]} for every valid state."""
    # Discover all valid states by iterating long_successor (covers V_n exactly).
    states = set(enumerate_longest(n))
    graph = {s: neighbors(s) for s in states}
    # sanity: every neighbor should itself be in states
    for s, nbrs in graph.items():
        for _, t in nbrs:
            assert t in states, f"neighbor {t} of {s} not in V_{n}"
    return graph


# ---------------------------------------------------------------------------
# Convenience: string <-> tuple
# ---------------------------------------------------------------------------

def state_from_string(s: str) -> State:
    return tuple(int(c) for c in s)


def state_to_string(w: State) -> str:
    return "".join(str(d) for d in w)
