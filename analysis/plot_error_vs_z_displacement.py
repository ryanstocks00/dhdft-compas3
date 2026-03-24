#!/usr/bin/env python3
"""
Signed isomerization-energy error (vs revDSD-PBEP86-D4) vs max_z_displacement.
Writes without-D4 and with-D4 PNGs (shared y-scale when both exist). Each figure
also has a *_small.png (1.65 in), styled like create_scatter_plot small figures.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from benchmark_compas3x import process_functional_comparison, process_xtb_comparison
from plotting_utils import SINGLE_COLUMN_WIDTH, format_axis_offsets

try:
    import scienceplots  # noqa: F401

    plt.style.use(["science", "ieee"])
except ImportError:
    pass

plt.rcParams.update(
    {
        "text.usetex": False,
        "mathtext.fontset": "stix",
        "mathtext.default": "regular",
        "font.serif": [
            "DejaVu Serif",
            "Liberation Serif",
            "Times New Roman",
            "Times",
            "serif",
        ],
        "font.monospace": [
            "DejaVu Sans Mono",
            "Liberation Mono",
            "Courier New",
            "monospace",
        ],
    }
)

FUNCTIONALS = [
    "SVWN5",
    "PBE",
    "BLYP",
    "revPBE",
    "BP86",
    "BPW91",
    "B97-D",
    "HCTH407",
    "TPSS",
    "MN15L",
    "SCAN",
    "rSCAN",
    "r2SCAN",
    "revTPSS",
    "t-HCTH",
    "M06-L",
    "M11-L",
]
MGGA_NO_D4 = ["M11-L", "MN15L", "t-HCTH"]


def _func_file_stem(func: str) -> str:
    return func.replace("-", "_").replace("(", "").replace(")", "").replace("/", "_")


def z_lookup_from_exess(df: pd.DataFrame) -> pd.DataFrame | None:
    if "max_z_displacement" not in df.columns:
        return None
    sub = df[
        (df["isomer_name"].str.contains("compas3x", case=False, na=False))
        & (df["optimizer"] == "GFN2-xTB")
    ].copy()
    from plotting_utils import extract_common_id

    sub["common_id"] = sub["isomer_name"].apply(extract_common_id)
    g = (
        sub.groupby("common_id", as_index=False)["max_z_displacement"]
        .first()
        .dropna(subset=["max_z_displacement"])
    )
    return g.set_index("common_id")


def _merge_err_z(res, ztab: pd.DataFrame) -> pd.DataFrame | None:
    m = res["merged"].copy()
    m["signed_err_kjmol"] = res["deviations"].values
    out = m.merge(ztab.reset_index(), on="common_id", how="inner")
    return out if len(out) >= 50 else None


def plot_signed_vs_max_z(
    signed_err: np.ndarray,
    max_z: np.ndarray,
    func_label: str,
    out_path: Path,
    ylim: tuple[float, float] | None = None,
    *,
    write_small: bool = True,
) -> None:
    def _mathtext_escape_label(s: str) -> str:
        return s.replace("-", r"\text{-}")

    variants: list[tuple[Path, tuple[float, float]]] = [
        (out_path, (SINGLE_COLUMN_WIDTH, 3.35)),
    ]
    if write_small:
        small_path = out_path.with_name(f"{out_path.stem}_small{out_path.suffix}")
        variants.append((small_path, (1.65, 1.65)))

    out_path.parent.mkdir(parents=True, exist_ok=True)

    for path, figsize in variants:
        size_scale = figsize[0] / SINGLE_COLUMN_WIDTH
        marker_size = max(2, int(6 * size_scale))
        linewidth_base = 1.0 * size_scale
        fontsize_label = max(6, int(8 * size_scale))
        # Slightly larger y-axis label on *_small figures (long two-line math reads tiny at ~1.65 in).
        fontsize_ylabel = (
            max(8, fontsize_label + 2) if size_scale < 0.6 else fontsize_label
        )
        fontsize_legend = max(5, int(7 * size_scale))
        labelpad = max(1, int(3 * size_scale)) if size_scale < 0.6 else 3
        # Tighter y than x so the (rotated) two-line label sits nearer the axis.
        labelpad_y = max(0, labelpad - 2)
        tick_labelsize = max(5, int(7 * size_scale))
        offset_text_fs = max(5, int(8 * size_scale))
        grid_lw = 0.5 * size_scale
        n_line_pts = 50 if size_scale >= 0.9 else 35

        fig, ax = plt.subplots(figsize=figsize)
        ax.axhline(0.0, color="gray", linewidth=linewidth_base, zorder=1)
        ax.scatter(
            max_z,
            signed_err,
            alpha=0.3,
            s=marker_size,
            c="#1f77b4",
            edgecolors="none",
            linewidth=0,
            zorder=2,
        )
        if len(max_z) > 10 and np.std(max_z) > 1e-9:
            m, b = np.polyfit(max_z, signed_err, 1)
            xs = np.linspace(float(np.min(max_z)), float(np.max(max_z)), n_line_pts)
            ccm = np.corrcoef(max_z, signed_err)
            r_lin = ccm[0, 1]
            r2 = float(r_lin**2) if np.isfinite(r_lin) else float("nan")
            fit_label = (
                rf"Linear fit ($r^2$ = {r2:.3f})" if np.isfinite(r2) else "Linear fit"
            )
            ax.plot(
                xs,
                m * xs + b,
                "k-",
                lw=1.0,
                alpha=0.7,
                label=fit_label,
                zorder=3,
            )
            ax.legend(
                fontsize=fontsize_legend,
                frameon=True,
                fancybox=False,
                edgecolor="black",
                loc="upper left",
            )
        ax.set_xlabel(
            r"Max $z$ displacement ($\mathrm{\AA}$)",
            fontsize=fontsize_label,
            labelpad=labelpad,
        )
        ylabel_line1 = (
            rf"$\Delta E^{{\mathrm{{{_mathtext_escape_label(func_label)}}}}} - "
            rf"\Delta E^{{\mathrm{{revDSD\text{{-}}PBEP86\text{{-}}D4}}}}$"
        )
        ylabel_line2 = r"$\mathrm{(kJ/mol)}$"
        ax.set_ylabel(
            f"{ylabel_line1}\n{ylabel_line2}",
            fontsize=fontsize_ylabel,
            labelpad=labelpad_y,
        )
        ax.tick_params(labelsize=tick_labelsize)
        ax.grid(True, alpha=0.3, linewidth=grid_lw)
        if ylim is not None:
            ax.set_ylim(ylim)

        if size_scale < 0.6:
            plt.subplots_adjust(left=0.18, bottom=0.18, right=0.98, top=0.98)
            pad_inches = 0.02
            # Keep vertical placement, move label left to avoid overlapping the y-axis spine.
            ax.yaxis.set_label_coords(-0.09, 0.46, transform=ax.transAxes)
        else:
            plt.tight_layout()
            pad_inches = 0.1

        format_axis_offsets(ax, fontsize=offset_text_fs)
        fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=pad_inches)
        plt.close(fig)
        print(f"Saved {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exess-csv", default="analysis/exess_data.csv")
    ap.add_argument(
        "--output-dir",
        default="plots",
        help="Directory relative to analysis/ (default: plots)",
    )
    ap.add_argument(
        "--reference-method",
        choices=["min", "avg", "linear_fit"],
        default="min",
    )
    args = ap.parse_args()

    script_dir = Path(__file__).resolve().parent
    root = script_dir.parent
    if Path(args.exess_csv).is_absolute():
        csv_path = Path(args.exess_csv)
    else:
        csv_path = root / args.exess_csv
    if Path(args.output_dir).is_absolute():
        out_dir = Path(args.output_dir)
    else:
        out_dir = script_dir / args.output_dir

    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    ztab = z_lookup_from_exess(df)
    if ztab is None or len(ztab) == 0:
        print(
            "WARNING: No max_z_displacement in EXESS CSV. "
            "Run: python analysis/extract_exess_data.py — skipping."
        )
        return

    fstem = _func_file_stem

    for functional in FUNCTIONALS:
        has_d4 = functional not in ("SVWN5", "HCTH407") and functional not in MGGA_NO_D4

        res_no = process_functional_comparison(
            df,
            functional,
            include_d4=False,
            output_dir=None,
            reference_method=args.reference_method,
            quiet=True,
        )
        if not res_no:
            continue
        m_no = _merge_err_z(res_no, ztab)
        if m_no is None:
            print(f"Skip {functional} (no D4): insufficient z merge")
            continue

        if has_d4:
            res_yes = process_functional_comparison(
                df,
                functional,
                include_d4=True,
                output_dir=None,
                reference_method=args.reference_method,
                quiet=True,
            )
            m_yes = _merge_err_z(res_yes, ztab) if res_yes else None
            errs = [m_no["signed_err_kjmol"].values]
            if m_yes is not None:
                errs.append(m_yes["signed_err_kjmol"].values)
            e = np.concatenate(errs)
            pad = 0.05 * (e.max() - e.min() + 1e-6)
            ylim = (float(e.min() - pad), float(e.max() + pad))
            plot_signed_vs_max_z(
                m_no["signed_err_kjmol"].values,
                m_no["max_z_displacement"].values,
                func_label=functional,
                out_path=out_dir
                / f"compas3x_{fstem(functional)}_without_d4_error_vs_max_z.png",
                ylim=ylim,
            )
            if m_yes is not None:
                plot_signed_vs_max_z(
                    m_yes["signed_err_kjmol"].values,
                    m_yes["max_z_displacement"].values,
                    func_label=f"{functional}-D4",
                    out_path=out_dir
                    / f"compas3x_{fstem(functional)}_with_d4_error_vs_max_z.png",
                    ylim=ylim,
                )
        else:
            e = m_no["signed_err_kjmol"].values
            pad = 0.05 * (e.max() - e.min() + 1e-6)
            ylim = (float(e.min() - pad), float(e.max() + pad))
            plot_signed_vs_max_z(
                e,
                m_no["max_z_displacement"].values,
                func_label=functional,
                out_path=out_dir
                / f"compas3x_{fstem(functional)}_without_d4_error_vs_max_z.png",
                ylim=ylim,
            )

    _buf = io.StringIO()
    with contextlib.redirect_stdout(_buf):
        xtb = process_xtb_comparison(
            df, output_dir=None, reference_method=args.reference_method
        )
    if xtb and "common_id" in xtb["merged"].columns:
        m = xtb["merged"].copy()
        m["signed_err_kjmol"] = xtb["deviations"].values
        m2 = m.merge(ztab.reset_index(), on="common_id", how="inner")
        if len(m2) >= 50:
            e = m2["signed_err_kjmol"].values
            pad = 0.05 * (e.max() - e.min() + 1e-6)
            ylim = (float(e.min() - pad), float(e.max() + pad))
            plot_signed_vs_max_z(
                e,
                m2["max_z_displacement"].values,
                func_label="GFN2-xTB",
                out_path=out_dir / "compas3x_GFN2_xTB_error_vs_max_z.png",
                ylim=ylim,
            )


if __name__ == "__main__":
    main()
