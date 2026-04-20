#!/usr/bin/env python3
"""Quick ball-and-stick preview from XYZ (matplotlib only)."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def read_xyz(path: Path) -> tuple[list[str], np.ndarray]:
    lines = path.read_text().strip().splitlines()
    n = int(lines[0])
    coords = []
    symbols = []
    for line in lines[2 : 2 + n]:
        parts = line.split()
        symbols.append(parts[0])
        coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return symbols, np.array(coords, dtype=float)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("xyz", type=Path)
    ap.add_argument("-o", "--output", type=Path, default=None)
    ap.add_argument("--bond-max", type=float, default=1.58, help="Max bond length (Å)")
    args = ap.parse_args()

    sym, pos = read_xyz(args.xyz)
    out = args.output or args.xyz.with_suffix(".png")

    radii = {"C": 0.35, "H": 0.22}
    colors = {"C": "#404040", "H": "#e8e8e8"}

    fig = plt.figure(figsize=(8, 8), facecolor="white")
    ax = fig.add_subplot(111, projection="3d")

    # Bonds
    n = len(sym)
    for i in range(n):
        for j in range(i + 1, n):
            d = float(np.linalg.norm(pos[i] - pos[j]))
            if d < args.bond_max:
                ax.plot3D(
                    [pos[i, 0], pos[j, 0]],
                    [pos[i, 1], pos[j, 1]],
                    [pos[i, 2], pos[j, 2]],
                    color="#888888",
                    linewidth=1.2,
                    zorder=1,
                )

    for s, p in zip(sym, pos):
        ax.scatter(
            [p[0]],
            [p[1]],
            [p[2]],
            s=(radii.get(s, 0.3) * 450) ** 2 / 25,
            c=colors.get(s, "#3366cc"),
            edgecolors="black",
            linewidths=0.2,
            zorder=2,
        )

    ax.set_title(args.xyz.stem, fontsize=11)
    ax.set_axis_off()
    # Equal aspect
    max_range = (pos.max(axis=0) - pos.min(axis=0)).max() / 2.0
    mid = (pos.max(axis=0) + pos.min(axis=0)) / 2.0
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)
    ax.view_init(elev=18, azim=72)
    plt.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
