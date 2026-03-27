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
from plotting_utils import (
    SINGLE_COLUMN_WIDTH,
    format_axis_offsets,
    format_linear_eq_3sf,
)

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


def linear_fit_signed_err_vs_max_z(
    max_z: np.ndarray, signed_err: np.ndarray
) -> tuple[float, float, float] | None:
    """OLS $y = m x + b$ with $x$ = max $z$ (Å), $y$ = signed error (kJ/mol).

    Returns ``(m, b, r^2)`` or ``None`` if a fit is not defined.
    """
    mz = np.asarray(max_z, dtype=float)
    se = np.asarray(signed_err, dtype=float)
    if len(mz) <= 10 or float(np.std(mz)) <= 1e-9:
        return None
    m, b = np.polyfit(mz, se, 1)
    ccm = np.corrcoef(mz, se)
    r_lin = ccm[0, 1]
    r2 = float(r_lin**2) if np.isfinite(r_lin) else float("nan")
    return float(m), float(b), r2


def _fit_to_summary_pair(
    fit: tuple[float, float, float] | None,
) -> tuple[float, float, float] | None:
    if fit is None:
        return None
    m, b, r2 = fit
    return (r2, m, b)


def print_z_displacement_fit_summary_table(
    rows: list[
        tuple[str, tuple[float, float, float] | None, tuple[float, float, float] | None]
    ],
) -> None:
    """Print ``functional``, $r^2$, gradient, intercept without D4, and with D4 (if applicable)."""
    print("\n" + "=" * 120)
    print(
        "Summary: signed isomerization-energy error vs max z displacement (linear fit)"
    )
    print("=" * 120)
    print(
        f"{'functional':<14}"
        f"{'r^2 (no D4)':>14}"
        f"{'grad (no D4)':>16}"
        f"{'int (no D4)':>16}"
        f"{'r^2 (D4)':>14}"
        f"{'grad (D4)':>16}"
        f"{'int (D4)':>16}"
    )
    print("-" * 120)

    def _fmt_r2_col(p: tuple[float, float, float] | None) -> str:
        if p is None:
            return f"{'—':>14}"
        r2, _m, _b = p
        if not np.isfinite(r2):
            return f"{'nan':>14}"
        return f"{r2:>14.4f}"

    def _fmt_grad_col(p: tuple[float, float, float] | None) -> str:
        if p is None:
            return f"{'—':>16}"
        _r2, m, _b = p
        return f"{m:>16.2f}"

    def _fmt_int_col(p: tuple[float, float, float] | None) -> str:
        if p is None:
            return f"{'—':>16}"
        _r2, _m, b = p
        return f"{b:>16.2f}"

    for func, no_d4, with_d4 in rows:
        print(
            f"{func:<14}"
            f"{_fmt_r2_col(no_d4)}"
            f"{_fmt_grad_col(no_d4)}"
            f"{_fmt_int_col(no_d4)}"
            f"{_fmt_r2_col(with_d4)}"
            f"{_fmt_grad_col(with_d4)}"
            f"{_fmt_int_col(with_d4)}"
        )
    print("=" * 120)
    print(
        "Gradients are in kJ/mol/AA and intercepts are in kJ/mol. "
        "Rows without a D4 variant use — in the D4 columns.\n"
    )


def write_z_displacement_fit_summary_latex_table(
    output_path: Path,
    rows: list[
        tuple[str, tuple[float, float, float] | None, tuple[float, float, float] | None]
    ],
    *,
    caption_prefix: str = "",
    label: str = "tab:compas3x_error_vs_maxz_fit",
) -> None:
    """Write the summary as a LaTeX table matching `compas3x_benchmarks.tex` style."""

    def _latex_func_name(func: str) -> str:
        return func.replace("-", "--")

    def _fmt_r2(p: tuple[float, float] | None) -> str:
        if p is None:
            return "---"
        r2, _m, _b = p
        return f"{r2:.3f}" if np.isfinite(r2) else "---"

    def _fmt_grad(p: tuple[float, float, float] | None) -> str:
        if p is None:
            return "---"
        _r2, m, _b = p
        return f"{m:.2f}"

    def _fmt_int(p: tuple[float, float, float] | None) -> str:
        if p is None:
            return "---"
        _r2, _m, b = p
        return f"{b:.2f}"

    # Match the same functional-category groupings as the SI benchmark table.
    lda_functionals = ["SVWN5"]
    gga_functionals = ["PBE", "BLYP", "revPBE", "BP86", "BPW91", "B97-D", "HCTH407"]
    mgga_functionals = [
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

    rows_map: dict[
        str,
        tuple[
            tuple[float, float, float] | None,
            tuple[float, float, float] | None,
        ],
    ] = {func: (no_d4, with_d4) for func, no_d4, with_d4 in rows}

    def _category_col(category: str, remaining: int) -> str:
        rot = f"\\rotatebox{{90}}{{\\emph{{{category}}}}}"
        # This matches the LDA 1-row styling used in compas3x_benchmarks.tex.
        if remaining == 1:
            inner = f"\\raisebox{{-0.5\\height}}{{{rot}}}"
            return f"\\multirow{{{remaining}}}{{*}}{{{inner}}}"
        return f"\\multirow{{{remaining}}}{{*}}{{{rot}}}"

    def _rcell(s: str) -> str:
        return f"\\raisebox{{-0.5\\height}}{{{s}}}"

    categories: list[tuple[str, list[str]]] = [
        ("LDA", lda_functionals),
        ("GGA", gga_functionals),
        ("MGGA", mgga_functionals),
    ]
    category_counts = {
        cat: sum(1 for fn in fns if fn in rows_map)
        for cat, fns in categories
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("% Auto-generated: error vs max_z_displacement fit summary\n")
        f.write("% Requires: \\usepackage{booktabs, multirow, rotating, graphicx}\n")
        f.write("\\begin{table}[H]\n")
        f.write("\\centering\n")
        f.write(
            "\\begin{tabular}{@{}c@{\\hspace{0.8em}}l@{\\hspace{0.25em}}"
            "c@{\\hspace{0.25em}}c@{\\hspace{0.25em}}c@{\\hspace{0.25em}}"
            "c@{\\hspace{0.25em}}c@{\\hspace{0.25em}}c@{}}"
            "\n"
        )
        f.write("\\toprule\n")
        f.write(
            " & \\multirow{2}{*}{Functional} & \\multicolumn{3}{c}{Without D4} & "
            "\\multicolumn{3}{c}{With D4} \\\\\n"
        )
        f.write("\\cmidrule(lr){3-5} \\cmidrule(lr){6-8}\n")
        _r2_h = "\\parbox[t]{2em}{$r^2$\\vspace{0.5em}}"
        _grad_h = "\\shortstack{Gradient\\\\\\footnotesize(kJ/mol/AA)}"
        _int_h = "\\shortstack{Intercept\\\\\\footnotesize(kJ/mol)}"
        f.write(
            f" & & {_r2_h} & {_grad_h} & {_int_h} & {_r2_h} & {_grad_h} & {_int_h} \\\\\n"
        )
        f.write("\\midrule\n")

        # XTB row (no category column), matching compas3x_benchmarks.tex style.
        if "GFN2-xTB" in rows_map:
            xtb_no, _xtb_with = rows_map["GFN2-xTB"]
            f.write(
                f" & GFN2--xTB & --- & --- & --- & "
                f"{_fmt_r2(xtb_no)} & {_fmt_grad(xtb_no)} & {_fmt_int(xtb_no)} \\\\\n"
            )
            f.write("\\midrule\n")

        for category, fns in categories:
            remaining = category_counts.get(category, 0)
            if remaining <= 0:
                continue

            first = True
            for fn in fns:
                if fn not in rows_map:
                    continue
                no_d4, with_d4 = rows_map[fn]
                cat_cell = _category_col(category, remaining) if first else ""
                first = False
                remaining -= 1

                fn_cell = _latex_func_name(fn)
                if category == "LDA":
                    # Match the 1-row LDA raisebox styling in compas3x_benchmarks.tex.
                    f.write(
                        f"{cat_cell} & {_rcell(fn_cell)} & {_rcell(_fmt_r2(no_d4))} & "
                        f"{_rcell(_fmt_grad(no_d4))} & {_rcell(_fmt_int(no_d4))} & "
                        f"{_rcell(_fmt_r2(with_d4))} & {_rcell(_fmt_grad(with_d4))} & "
                        f"{_rcell(_fmt_int(with_d4))} \\\\[2ex]\n"
                    )
                else:
                    f.write(
                        f"{cat_cell} & {fn_cell} & {_fmt_r2(no_d4)} & {_fmt_grad(no_d4)} & {_fmt_int(no_d4)} & "
                        f"{_fmt_r2(with_d4)} & {_fmt_grad(with_d4)} & {_fmt_int(with_d4)} \\\\\n"
                    )
            f.write("\\midrule\n")

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        cap = (
            "Coefficient of determination ($r^2$) and linear-fit gradient for signed "
            "isomerization-energy error versus maximum $z$ displacement for COMPAS-3x "
            "geometries, split by inclusion of D4."
        )
        f.write(f"\\caption{{{caption_prefix}{cap}}}\n")
        f.write(f"\\label{{{label}}}\n")
        f.write("\\end{table}\n")


def plot_signed_vs_max_z(
    signed_err: np.ndarray,
    max_z: np.ndarray,
    func_label: str,
    out_path: Path,
    ylim: tuple[float, float] | None = None,
    *,
    fit: tuple[float, float, float] | None = None,
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
        # Slightly larger y-axis label on *_small figures (long label reads tiny at ~1.65 in).
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
        fit_res = (
            fit
            if fit is not None
            else linear_fit_signed_err_vs_max_z(
                np.asarray(max_z), np.asarray(signed_err)
            )
        )
        m_fit = b_fit = None
        eq_tex = None
        if fit_res is not None:
            m_fit, b_fit, r2_fit = fit_res
            x0_seg = float(np.min(max_z))
            x1_seg = float(np.max(max_z))
            xs = np.linspace(x0_seg, x1_seg, n_line_pts)
            ax.plot(
                xs,
                m_fit * xs + b_fit,
                "k-",
                lw=1.0,
                alpha=0.7,
                label=(
                    (
                        rf"Linear fit:"
                        f"\n{format_linear_eq_3sf(m_fit, b_fit)}"
                        rf" ($r^2$ = {r2_fit:.3f})"
                    )
                    if size_scale < 0.6
                    else (
                        rf"Linear fit: {format_linear_eq_3sf(m_fit, b_fit)}"
                        rf" ($r^2$ = {r2_fit:.3f})"
                    )
                ),
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
        units_fontsize = max(5, int(round(fontsize_ylabel * 0.8)))  # 20% smaller
        ax.tick_params(labelsize=tick_labelsize)
        ax.grid(True, alpha=0.3, linewidth=grid_lw)
        if ylim is not None:
            ax.set_ylim(ylim)

        if size_scale < 0.6:
            plt.subplots_adjust(left=0.18, bottom=0.18, right=0.98, top=0.98)
            pad_inches = 0.02
            # Small-plot placement tweaks: main label further left; units higher without extra left shift.
            ax.set_ylabel(
                ylabel_line1,
                fontsize=fontsize_ylabel,
                labelpad=labelpad_y,
            )
            ax.yaxis.set_label_coords(-0.17, 0.46, transform=ax.transAxes)
            units_x, units_y = -0.15, 0.45
            # Draw units separately for small plots so only this line is smaller.
            ax.text(
                units_x,
                units_y,
                "(kJ/mol)",
                transform=ax.transAxes,
                rotation=90,
                va="center",
                ha="center",
                fontsize=units_fontsize,
            )
        else:
            # Large plot: single-line y label (units on same line as the ΔE expression).
            ylabel_large = ylabel_line1[:-1] + r"\, \mathrm{(kJ/mol)}$"
            ax.set_ylabel(
                ylabel_large,
                fontsize=fontsize_ylabel,
                labelpad=labelpad_y,
            )
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
    summary_rows: list[
        tuple[str, tuple[float, float] | None, tuple[float, float] | None]
    ] = []

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
            fit_no = linear_fit_signed_err_vs_max_z(
                m_no["max_z_displacement"].values,
                m_no["signed_err_kjmol"].values,
            )
            fit_yes = (
                linear_fit_signed_err_vs_max_z(
                    m_yes["max_z_displacement"].values,
                    m_yes["signed_err_kjmol"].values,
                )
                if m_yes is not None
                else None
            )
            summary_rows.append(
                (
                    functional,
                    _fit_to_summary_pair(fit_no),
                    _fit_to_summary_pair(fit_yes),
                )
            )
            plot_signed_vs_max_z(
                m_no["signed_err_kjmol"].values,
                m_no["max_z_displacement"].values,
                func_label=functional,
                out_path=out_dir
                / f"compas3x_{fstem(functional)}_without_d4_error_vs_max_z.png",
                ylim=ylim,
                fit=fit_no,
            )
            if m_yes is not None:
                plot_signed_vs_max_z(
                    m_yes["signed_err_kjmol"].values,
                    m_yes["max_z_displacement"].values,
                    func_label=f"{functional}-D4",
                    out_path=out_dir
                    / f"compas3x_{fstem(functional)}_with_d4_error_vs_max_z.png",
                    ylim=ylim,
                    fit=fit_yes,
                )
        else:
            e = m_no["signed_err_kjmol"].values
            pad = 0.05 * (e.max() - e.min() + 1e-6)
            ylim = (float(e.min() - pad), float(e.max() + pad))
            fit_no = linear_fit_signed_err_vs_max_z(
                m_no["max_z_displacement"].values,
                m_no["signed_err_kjmol"].values,
            )
            summary_rows.append(
                (functional, _fit_to_summary_pair(fit_no), None)
            )
            plot_signed_vs_max_z(
                e,
                m_no["max_z_displacement"].values,
                func_label=functional,
                out_path=out_dir
                / f"compas3x_{fstem(functional)}_without_d4_error_vs_max_z.png",
                ylim=ylim,
                fit=fit_no,
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
            fit_xtb = linear_fit_signed_err_vs_max_z(
                m2["max_z_displacement"].values,
                m2["signed_err_kjmol"].values,
            )
            summary_rows.append(
                ("GFN2-xTB", _fit_to_summary_pair(fit_xtb), None)
            )
            plot_signed_vs_max_z(
                e,
                m2["max_z_displacement"].values,
                func_label="GFN2-xTB",
                out_path=out_dir / "compas3x_GFN2_xTB_error_vs_max_z.png",
                ylim=ylim,
                fit=fit_xtb,
            )

    print_z_displacement_fit_summary_table(summary_rows)
    write_z_displacement_fit_summary_latex_table(
        out_dir / "compas3x_error_vs_max_z_fit_summary.tex",
        summary_rows,
    )


if __name__ == "__main__":
    main()
