#!/usr/bin/env python3
"""Draw docs/figures/transpose-quadrants.png: how 2_transpose/snippet.c moves one 8x8 block.

The top row (Steps 1-3) follows the three k-loops of snippet.c. The bottom row is not
drawn by hand: this script replays snippet.c's loads and stores of A and B through a
Python port of the Tree-PLRU cache in 1_cachesim/cachesim.cc, using the geometry that
2_transpose/evaluate.c passes to csim (16 sets, 2 ways, 32-byte lines). It first checks
that the replay reproduces the miss counts in the README (1,152 / 272 for 32x32,
4,608 / 1,344 for 64x64), then draws the cache-set contents it observes.

Usage (from the repository root; needs matplotlib):

    python3 docs/figures/make_transpose_figure.py
"""

from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Rectangle  # noqa: E402

OUT = Path(__file__).with_name("transpose-quadrants.png")

# --------------------------------------------------------------------------------------
# 1. Replay: Tree-PLRU port + the access order of snippet.c
# --------------------------------------------------------------------------------------

SETS, WAYS, LINE = 16, 2, 32  # evaluate.c: ./csim_cpp -S 16 -W 2 -B 32
README_MISSES = {32: (1152, 272), 64: (4608, 1344)}  # README "Results": (baseline, mine)


class TreePLRU:
    """cache_sim_t::access / victimize / update_tree_on_hit from 1_cachesim/cachesim.cc."""

    def __init__(self, sets=SETS, ways=WAYS, linesz=LINE):
        self.sets, self.ways = sets, ways
        self.shift = linesz.bit_length() - 1
        self.tags = [None] * (sets * ways)  # None = invalid way
        self.tree = [0] * (sets * ways)  # plru_tree, zero-initialised
        self.misses = 0

    def set_of(self, addr):
        return (addr >> self.shift) & (self.sets - 1)

    def access(self, addr):
        """Return (hit, evicted_tag)."""
        base = self.set_of(addr) * self.ways
        tag = addr >> self.shift
        for way in range(self.ways):
            if self.tags[base + way] == tag:  # hit: point every parent away from this leaf
                node = self.ways + way
                while node > 1:
                    self.tree[base + node // 2] = int(node % 2 == 0)
                    node //= 2
                return True, None
        self.misses += 1  # miss: follow the bits from the root, flipping each one
        node = 1
        while node < self.ways:
            bit = self.tree[base + node]
            self.tree[base + node] = bit ^ 1
            node = 2 * node + bit
        way = node - self.ways
        victim = self.tags[base + way]
        self.tags[base + way] = tag
        return False, victim


def bases(n):
    """A and B are both 4 KiB aligned (trans-driver.c); only that affects set indices."""
    a_base = 0x10000
    return a_base, a_base + max(n * n * 4, 4096)


def snippet_accesses(n, step2_touches=True):
    """The A/B accesses of 2_transpose/snippet.c in program order.

    Yields (block, step, k, op, matrix, row, col); op is "L" (load), "S" (store) or
    "T" (one of the `l = B[..][..]` loads).
    """
    for i in range(0, n, 8):
        for j in range(0, n, 8):
            blk = (i, j)
            for k in range(i, i + 4):  # Step 1: for(k=i;k<i+4;k++)
                for c in range(8):
                    yield blk, 1, k, "L", "A", k, j + c
                for op, r, c in (("S", j, k), ("S", j + 1, k), ("T", j, k),
                                 ("S", j + 2, k), ("T", j + 1, k), ("S", j + 3, k),
                                 ("S", j, k + 4), ("S", j + 1, k + 4), ("T", j, k + 4),
                                 ("S", j + 2, k + 4), ("T", j + 1, k + 4), ("S", j + 3, k + 4)):
                    yield blk, 1, k, op, "B", r, c
            for k in range(j, j + 4):  # Step 2: for(k=j;k<j+4;k++)
                for c in range(4):
                    yield blk, 2, k, "L", "B", k, i + 4 + c
                for r in range(4):
                    yield blk, 2, k, "L", "A", i + 4 + r, k
                for c in range(4):
                    yield blk, 2, k, "S", "B", k, i + 4 + c
                if step2_touches:
                    r, c = {j: (j + 2, i + 4), j + 1: (j + 3, i + 4),
                            j + 2: (j + 4, i), j + 3: (j + 5, i)}[k]
                    yield blk, 2, k, "T", "B", r, c
                for c in range(4):
                    yield blk, 2, k, "S", "B", k + 4, i + c
            for k in range(i + 4, i + 8):  # Step 3: for(k=i+4;k<i+8;k++)
                for c in range(4):
                    yield blk, 3, k, "L", "A", k, j + 4 + c
                for op, r, c in (("S", j + 4, k), ("S", j + 5, k), ("T", j + 4, k),
                                 ("S", j + 6, k), ("T", j + 5, k), ("S", j + 7, k)):
                    yield blk, 3, k, op, "B", r, c


def naive_accesses(n):
    """trans_naive() in trans-driver.c."""
    for i in range(n):
        for j in range(n):
            yield None, 0, i, "L", "A", i, j
            yield None, 0, i, "S", "B", j, i


def addr(n, matrix, row, col):
    a_base, b_base = bases(n)
    return (a_base if matrix == "A" else b_base) + (row * n + col) * 4


def replay(n, accesses, watch=None):
    """Run the accesses; return (cache, misses per (block, step), events in set `watch`)."""
    cache = TreePLRU()
    per_step = Counter()
    events = []
    for blk, step, k, op, m, r, c in accesses:
        a = addr(n, m, r, c)
        hit, victim = cache.access(a)
        if not hit:
            per_step[blk, step] += 1
        if watch is not None and blk == watch[0] and cache.set_of(a) == watch[1]:
            base = watch[1] * WAYS
            events.append(dict(step=step, k=k, op=op, row=r, hit=hit, victim=victim,
                               ways=list(cache.tags[base:base + WAYS])))
    return cache, per_step, events


def b_row_of(n, tag):
    _, b_base = bases(n)
    a = tag << (LINE.bit_length() - 1)
    assert a >= b_base, "expected only B lines in this set"
    return (a - b_base) // (n * 4)


# Check the port against the judge numbers quoted in the README.
for n, (naive, mine) in README_MISSES.items():
    assert replay(n, naive_accesses(n))[0].misses == naive, n
    assert replay(n, snippet_accesses(n))[0].misses == mine, n
    # Every off-diagonal block misses once per line: 8 A lines + 8 B lines.
    per_step = replay(n, snippet_accesses(n))[1]
    for bi in range(0, n, 8):
        for bj in range(0, n, 8):
            if bi != bj:
                got = [per_step[(bi, bj), s] for s in (1, 2, 3)]
                assert got == [8, 8, 0], (n, bi, bj, got)

# Representative off-diagonal block of the 64x64 case; set s holds B rows j, j+2, j+4, j+6.
N, BI, BJ = 64, 16, 8
_, B_BASE = bases(N)
S = TreePLRU().set_of(addr(N, "B", BJ, BI))
assert all(TreePLRU().set_of(addr(N, "B", BJ + r, BI)) == S for r in (0, 2, 4, 6))
_, _, ev = replay(N, snippet_accesses(N), watch=((BI, BJ), S))


def rel(tag):
    r = b_row_of(N, tag) - BJ
    return "row j" if r == 0 else f"row j+{r}"


def snapshot(e):
    return [rel(t) for t in e["ways"]]


end1 = [e for e in ev if e["step"] == 1][-1]
miss4 = next(e for e in ev if e["step"] == 2 and not e["hit"] and e["row"] == BJ + 4)
miss6 = next(e for e in ev if e["step"] == 2 and not e["hit"] and e["row"] == BJ + 6)
end3 = ev[-1]
assert snapshot(end1) == ["row j", "row j+2"]
assert rel(miss4["victim"]) == "row j" and miss4["k"] == BJ
assert rel(miss6["victim"]) == "row j+2" and miss6["k"] == BJ + 2
assert not any(not e["hit"] for e in ev if e["step"] == 3)
# Counterfactual: drop the four `l = B[..][..]` touches of Step 2.
_, _, ev_nt = replay(N, snippet_accesses(N, step2_touches=False), watch=((BI, BJ), S))
miss4_nt = next(e for e in ev_nt if e["step"] == 2 and not e["hit"] and e["row"] == BJ + 4)
assert rel(miss4_nt["victim"]) == "row j+2"

SNAPSHOTS = [
    ("End of Step 1", snapshot(end1), None, "Both lines were\nfilled in Step 1."),
    ("Step 2, k = j", snapshot(miss4), 0, "Touch row j+2;\nrow j+4 then evicts\nrow j (finished)."),
    ("Step 2, k = j+2", snapshot(miss6), 1, "Touch row j+4;\nrow j+6 then evicts\nrow j+2 (finished)."),
    ("Step 3", snapshot(end3), None, "Rows j+4, j+6 still\ncached: no misses."),
]

# --------------------------------------------------------------------------------------
# 2. Drawing
# --------------------------------------------------------------------------------------

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#1f2328", "#57606a", "#8c959f"
DONE, PANEL, BG = "#e3e6ea", "#f6f8fa", "#ffffff"
SET_S, SET_S8 = "#6e7781", "#d0d7de"
MONO = "DejaVu Sans Mono"

plt.rcParams.update({"font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans"})


def tint(hex_color, amount):
    """Blend a colour with white; amount = share of the colour."""
    rgb = [int(hex_color[p:p + 2], 16) for p in (1, 3, 5)]
    return "#" + "".join(f"{round(255 - (255 - v) * amount):02x}" for v in rgb)


W, H = 1200, 950
fig = plt.figure(figsize=(W / 100, H / 100), dpi=100, facecolor=BG)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(H, 0)
ax.axis("off")
RENDERER = fig.canvas.get_renderer()


def width(s, size, **kw):
    """Rendered width of a string in pixels (= data units here)."""
    t = ax.text(0, 0, s, fontsize=size, **kw)
    w = t.get_window_extent(RENDERER).width
    t.remove()
    return w


def text(x, y, s, size=12, color=INK, **kw):
    kw.setdefault("va", "top")
    ax.text(x, y, s, fontsize=size, color=color, **kw)


def panel(x, y, w, h):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=PANEL, edgecolor="#d0d7de", lw=1))


QUAD = {"11": (0, 0), "12": (0, 4), "21": (4, 0), "22": (4, 4)}
STYLE = {
    "read_blue": dict(face=tint(BLUE, 0.16), edge=BLUE, ls=(0, (3, 2)), hatch=None),
    "read_orange": dict(face=tint(ORANGE, 0.16), edge=ORANGE, ls=(0, (3, 2)), hatch=None),
    "final": dict(face=tint(BLUE, 0.55), edge=BLUE, ls="-", hatch=None),
    "parked": dict(face=tint(ORANGE, 0.45), edge=ORANGE, ls="-", hatch="///"),
    "done": dict(face=DONE, edge=GRID, ls="-", hatch=None),
    "idle": dict(face=BG, edge=GRID, ls="-", hatch=None),
}


def block(x, y, cell, name, rows, quads, labels):
    """8x8 block; every row of cells is one 32-byte cache line."""
    for q, (r0, c0) in QUAD.items():
        st = STYLE[quads.get(q, "idle")]
        ax.add_patch(Rectangle((x + c0 * cell, y + r0 * cell), 4 * cell, 4 * cell,
                               facecolor=st["face"], edgecolor="none", hatch=st["hatch"]))
        if st["hatch"]:
            ax.add_patch(Rectangle((x + c0 * cell, y + r0 * cell), 4 * cell, 4 * cell,
                                   facecolor="none", edgecolor=ORANGE, lw=0, hatch="///"))
    for t in range(9):  # thin cell lines
        ax.plot([x, x + 8 * cell], [y + t * cell] * 2, color=GRID, lw=0.5, alpha=0.6)
        ax.plot([x + t * cell] * 2, [y, y + 8 * cell], color=GRID, lw=0.5, alpha=0.6)
    for q, (r0, c0) in QUAD.items():  # quadrant outlines in the state colour
        st = STYLE[quads.get(q, "idle")]
        ax.add_patch(Rectangle((x + c0 * cell + 1, y + r0 * cell + 1), 4 * cell - 2,
                               4 * cell - 2, facecolor="none", edgecolor=st["edge"],
                               lw=1.8, ls=st["ls"]))
    ax.add_patch(Rectangle((x, y), 8 * cell, 8 * cell, facecolor="none", edgecolor=INK, lw=1.4))
    text(x + 4 * cell, y - 8, name, size=12, ha="center", va="bottom")
    text(x - 6, y + 0.5 * cell, rows[0], size=10.5, color=MUTED, ha="right", va="center",
         family=MONO)
    text(x - 6, y + 4.5 * cell, rows[1], size=10.5, color=MUTED, ha="right", va="center",
         family=MONO)
    for q, (s, dy) in labels.items():
        r0, c0 = QUAD[q]
        state = quads.get(q, "idle")
        col = MUTED if state in ("done", "idle") else INK
        ax.text(x + (c0 + 2) * cell, y + (r0 + 2) * cell + dy, s, fontsize=13, color=col,
                ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.15", fc=BG, ec="none", alpha=0.85)
                if state in ("parked", "final") else None)


def arrow(p, q, color, rad=0.0, lw=2.2, style="-|>", ls="-"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=16, color=color,
                                 lw=lw, ls=ls, connectionstyle=f"arc3,rad={rad}",
                                 shrinkA=0, shrinkB=0, zorder=5))


# ---- header ---------------------------------------------------------------------------
text(24, 18, "How snippet.c transposes one 8×8 block (B = Aᵀ)", size=19, weight="bold")
text(24, 52, "Each row of a block is one 32-byte cache line (8 ints). "
     "Quadrants: 11 top-left, 12 top-right, 21 bottom-left, 22 bottom-right.",
     size=12, color=MUTED)

lx, ly = 24, 86
for styles, label in ((("final",), "written to its final place"),
                      (("parked",), "parked in B's top-right, moved in Step 2"),
                      (("read_blue", "read_orange"), "read in this step"),
                      (("done",), "finished in an earlier step")):
    for n_sw, style in enumerate(styles):
        st = STYLE[style]
        ax.add_patch(Rectangle((lx, ly), 22, 16, facecolor=st["face"], edgecolor=st["edge"],
                               lw=1.6, ls=st["ls"], hatch=st["hatch"]))
        lx += 28
    text(lx + 2, ly + 8, label, size=11.5, va="center")
    lx += 2 + width(label, 11.5) + 30

# ---- step panels ----------------------------------------------------------------------
PX, PY, PW, PH, GAP = 20, 122, 373, 372, 20
CELL = 17
steps = [
    dict(title="Step 1", code="for (k = i; k < i+4; k++)",
         a=dict(quads={"11": "read_blue", "12": "read_orange"},
                labels={"11": ("$A_{11}$", 0), "12": ("$A_{12}$", 0),
                        "21": ("$A_{21}$", 0), "22": ("$A_{22}$", 0)}),
         b=dict(quads={"11": "final", "12": "parked"},
                labels={"11": ("$A_{11}^{T}$", 0), "12": ("$A_{12}^{T}$", 0)}),
         body="Load row k of A (one line).\n"
              "Left half → column k of $B_{11}$.\n"
              "Right half → column k of $B_{12}$,\n"
              "parked in the same B lines\n(rows j … j+3)."),
    dict(title="Step 2", code="for (k = j; k < j+4; k++)",
         a=dict(quads={"11": "done", "12": "done", "21": "read_blue"},
                labels={"21": ("$A_{21}$", 0), "22": ("$A_{22}$", 0)}),
         b=dict(quads={"11": "done", "12": "final", "21": "final"},
                labels={"11": ("$A_{11}^{T}$", 0), "12": ("$A_{21}^{T}$", -12),
                        "21": ("$A_{12}^{T}$", 12)}),
         body="Row k of $B_{12}$: read the parked\n"
              "values, overwrite with column k\nof $A_{21}$. "
              "Then write the parked\nvalues into row k+4 of $B_{21}$.",
         swap=True),
    dict(title="Step 3", code="for (k = i+4; k < i+8; k++)",
         a=dict(quads={"11": "done", "12": "done", "21": "done", "22": "read_blue"},
                labels={"22": ("$A_{22}$", 0)}),
         b=dict(quads={"11": "done", "12": "done", "21": "done", "22": "final"},
                labels={"22": ("$A_{22}^{T}$", 0)}),
         body="Row k of $A_{22}$ → column k of $B_{22}$.\n"
              "A rows i+4 … i+7 and B rows\n"
              "j+4 … j+7 were already loaded\nin Step 2."),
]
for n_step, st in enumerate(steps):
    x0 = PX + n_step * (PW + GAP)
    panel(x0, PY, PW, PH)
    text(x0 + 16, PY + 14, st["title"], size=15, weight="bold")
    text(x0 + 16 + width(st["title"], 15, weight="bold") + 10, PY + 17, st["code"],
         size=11, family=MONO, color=MUTED)
    gy = PY + 80
    ax_x, bx_x = x0 + 44, x0 + 44 + 8 * CELL + 50
    block(ax_x, gy, CELL, "A block", ("i", "i+4"), st["a"]["quads"], st["a"]["labels"])
    block(bx_x, gy, CELL, "B block", ("j", "j+4"), st["b"]["quads"], st["b"]["labels"])
    arrow((ax_x + 8 * CELL + 7, gy + 2.5 * CELL), (bx_x - 8, gy + 2.5 * CELL), INK, lw=1.8)
    if st.get("swap"):  # parked values: B12 -> B21
        arrow((bx_x + 5.2 * CELL, gy + 2.9 * CELL), (bx_x + 2.8 * CELL, gy + 5.1 * CELL),
              ORANGE, rad=0.25, lw=2.6)
    text(x0 + 16, gy + 8 * CELL + 20, st["body"], size=12, linespacing=1.45)

# ---- bottom panel: why the order matters ----------------------------------------------
BY, BH = PY + PH + 20, H - (PY + PH + 20) - 20
panel(PX, BY, W - 2 * PX, BH)
text(PX + 16, BY + 14, "Why this order avoids conflict misses", size=15, weight="bold")
text(PX + 16, BY + 40, "64×64 case, cache of 16 sets × 2 ways × 32 B. "
     "Replayed with the Tree-PLRU from Part 1 on an off-diagonal block.",
     size=12, color=MUTED)

# left: row -> set mapping of the B block
rx, ry, rw, rh = PX + 108, BY + 104, 150, 20
text(rx + rw / 2, ry - 10, "B block rows", size=12, ha="center", va="bottom")
for r in range(8):
    col = SET_S if r % 2 == 0 else SET_S8
    ax.add_patch(Rectangle((rx, ry + r * (rh + 3)), rw, rh, facecolor=col, edgecolor="none"))
    text(rx - 8, ry + r * (rh + 3) + rh / 2, "row j" if r == 0 else f"row j+{r}", size=11,
         family=MONO, ha="right", va="center")
    text(rx + rw + 8, ry + r * (rh + 3) + rh / 2, "set s" if r % 2 == 0 else "set s+8",
         size=11, family=MONO, ha="left", va="center", color=MUTED)
text(PX + 16, ry + 8 * (rh + 3) + 12,
     "A 64-int row is 256 B, so rows two apart\n"
     "share a set. The 8 rows of a B block get\n"
     "2 sets × 2 ways = room for 4 lines, so only\n"
     "half of the block can be cached at a time.",
     size=12, linespacing=1.45)

# right: contents of set s over time, straight from the replay
sx0, sy = PX + 400, BY + 92
text(sx0, sy - 12, "Set s (B rows j, j+2, j+4, j+6) as the steps run", size=12,
     va="bottom")
slot_w, slot_h, snap_gap = 82, 30, 18
for n_snap, (title, ways, new_way, note) in enumerate(SNAPSHOTS):
    x = sx0 + n_snap * (2 * slot_w + snap_gap)
    text(x, sy + 6, title, size=12, weight="bold")
    for w_i, label in enumerate(ways):
        fresh = (w_i == new_way)
        ax.add_patch(Rectangle((x + w_i * slot_w, sy + 32), slot_w - 4, slot_h,
                               facecolor=tint(BLUE, 0.55) if fresh else BG,
                               edgecolor=BLUE if fresh else GRID, lw=1.6))
        text(x + w_i * slot_w + (slot_w - 4) / 2, sy + 32 + slot_h / 2, label, size=11,
             family=MONO, ha="center", va="center")
    text(x, sy + 32 + slot_h + 6, "way 0", size=9.5, color=MUTED)
    text(x + slot_w, sy + 32 + slot_h + 6, "way 1", size=9.5, color=MUTED)
    text(x, sy + 94, note, size=11.5, linespacing=1.4)
    if n_snap < len(SNAPSHOTS) - 1:
        arrow((x + 2 * slot_w - 2, sy + 32 + slot_h / 2),
              (x + 2 * slot_w + snap_gap - 2, sy + 32 + slot_h / 2), MUTED, lw=1.4)

text(sx0, sy + 180,
     "Why the touch: at k = j, l = B[j+2][i+4] makes row j+2 the most recently used\n"
     "line. Without it, row j (just written) would be the most recent, and row j+4\n"
     "would evict row j+2, which k = j+2 still needs.",
     size=12, linespacing=1.45)
text(sx0, sy + 262,
     "Result: every off-diagonal block misses exactly once per line:\n"
     "8 misses in Step 1, 8 in Step 2 and 0 in Step 3, for both 32×32 and 64×64.",
     size=12, linespacing=1.45, weight="bold")

fig.savefig(OUT, dpi=100, facecolor=BG)
print(f"wrote {OUT} ({W}x{H}px); replay matches README miss counts {README_MISSES}")
