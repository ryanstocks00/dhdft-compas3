"""Common plotting utilities for comparison scripts."""

import numpy as np
import matplotlib.pyplot as plt

# Try to import scienceplots, but continue without it if not available
try:
    import scienceplots
    plt.style.use(['science', 'ieee'])
except ImportError:
    pass

# Matplotlib configuration
plt.rcParams.update({
    'text.usetex': False, 'mathtext.fontset': 'stix', 'mathtext.default': 'regular',
    'font.serif': ['DejaVu Serif', 'Liberation Serif', 'Times New Roman', 'Times', 'serif'],
    'font.monospace': ['DejaVu Sans Mono', 'Liberation Mono', 'Courier New', 'monospace']
})

# Constants
HARTREE_TO_KJ_PER_MOL = 2625.5
SINGLE_COLUMN_WIDTH = 3.5


def format_functional_name(functional):
    """Format functional name for display in plots (e.g., r2SCAN -> r²SCAN)."""
    # Replace r2SCAN with superscript 2
    if functional == 'r2SCAN':
        return r'r$^2$SCAN'
    return functional


def extract_common_id(name):
    """Extract common identifier from isomer name."""
    if not isinstance(name, str):
        return name
    parts = name.split('_', 1)
    if len(parts) > 1:
        common_id = parts[1]
        return common_id[3:] if common_id.startswith('hc_') else common_id
    return name


def calculate_stats(x, y):
    """Calculate r², RMSD, and MAD statistics."""
    correlation = np.corrcoef(x, y)[0, 1]
    r_squared = correlation ** 2
    rmsd = np.sqrt(np.mean((x - y) ** 2))
    abs_errors, abs_x = np.abs(x - y), np.abs(x)
    # np.where still evaluates both branches; use divide(..., where=) to avoid warnings.
    rel = np.zeros_like(abs_errors, dtype=float)
    np.divide(abs_errors, abs_x, out=rel, where=(abs_x > 0))
    mad_percentage = float(np.mean(rel) * 100)
    return r_squared, rmsd, mad_percentage


def format_axis_offsets(ax, fontsize=8):
    """Move axis offset text to corners."""
    for axis, pos, ha, va in [(ax.xaxis, (0.98, 0.98), 'right', 'top'), (ax.yaxis, (0.02, 0.02), 'left', 'bottom')]:
        offset = axis.get_offset_text()
        if offset.get_text():
            offset.set_visible(False)
            ax.text(pos[0], pos[1], offset.get_text(), transform=ax.transAxes,
                   horizontalalignment=ha, verticalalignment=va, fontsize=fontsize, color=offset.get_color())


def create_scatter_plot(x, y, xlabel, ylabel, output_path, mad_kjmol=None, msd_kjmol=None, figsize=None, xlim=None, ylim=None, show_linear_fits=False):
    """Create a standardized scatter plot with statistics.
    
    Args:
        x: x-axis data
        y: y-axis data
        xlabel: x-axis label
        ylabel: y-axis label
        output_path: path to save the plot
        mad_kjmol: Mean Absolute Deviation in kJ/mol (optional, will be calculated if not provided)
        msd_kjmol: Mean Signed Deviation in kJ/mol (optional, will be calculated if not provided)
        figsize: tuple of (width, height) in inches (optional, defaults to SINGLE_COLUMN_WIDTH x SINGLE_COLUMN_WIDTH)
        xlim: tuple of (xmin, xmax) for x-axis limits (optional, auto-calculated if not provided)
        ylim: tuple of (ymin, ymax) for y-axis limits (optional, auto-calculated if not provided)
        show_linear_fits: whether to show linear fit lines (default: False)
    """
    from pathlib import Path
    
    if figsize is None:
        figsize = (SINGLE_COLUMN_WIDTH, SINGLE_COLUMN_WIDTH)
    fig, ax = plt.subplots(figsize=figsize)
    
    # Scale sizes proportionally to figure size
    size_scale = figsize[0] / SINGLE_COLUMN_WIDTH
    marker_size = max(2, int(6 * size_scale))
    linewidth_base = 1.0 * size_scale
    # Font sizes scaled for small plots (slightly increased from previous version)
    fontsize_label = max(6, int(8 * size_scale))
    fontsize_legend = max(5, int(7 * size_scale))
    fontsize_text = max(5, int(7 * size_scale))
    bbox_linewidth = 0.5 * size_scale

    x_arr = np.asarray(x)
    y_arr = np.asarray(y)
    abs_errors = np.abs(y_arr - x_arr)
    max_err_idx = int(np.argmax(abs_errors))
    max_err_x = float(x_arr[max_err_idx])
    max_err_y = float(y_arr[max_err_idx])
    max_err_val = float(abs_errors[max_err_idx])
    
    ax.scatter(x_arr, y_arr, alpha=0.3, s=marker_size, color='#1f77b4', edgecolors='none', linewidth=0)
    ax.scatter([max_err_x], [max_err_y], s=marker_size * 2.2, color='#d62728',
               edgecolors='none', linewidth=0, zorder=6)
    
    min_val, max_val = min(float(x_arr.min()), float(y_arr.min())), max(float(x_arr.max()), float(y_arr.max()))
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=linewidth_base, label='Perfect agreement')
    
    if show_linear_fits and len(x_arr) > 1:
        # x->y fit: y = slope_xy * x + intercept_xy
        slope_xy, intercept_xy = np.polyfit(x_arr, y_arr, 1)
        trendline_x = np.array([min_val, max_val])
        ax.plot(trendline_x, slope_xy * trendline_x + intercept_xy, 'black', alpha=0.8, linewidth=0.9 * size_scale,
               linestyle='-', label='Linear fit (x→y)', zorder=10)
        
        # y->x fit: x = slope_yx * y + intercept_yx, so y = (x - intercept_yx) / slope_yx
        slope_yx, intercept_yx = np.polyfit(y_arr, x_arr, 1)
        # Invert to plot: y = (x - intercept_yx) / slope_yx
        ax.plot(trendline_x, (trendline_x - intercept_yx) / slope_yx, 'gray', alpha=0.8, linewidth=0.9 * size_scale,
               linestyle='--', label='Linear fit (y→x)', zorder=10)
    
    r_squared, rmsd, mad_percentage = calculate_stats(x_arr, y_arr)
    if mad_kjmol is None:
        mad_kjmol = np.mean(np.abs(x_arr - y_arr))
    if msd_kjmol is None:
        msd_kjmol = np.mean(y_arr - x_arr)
    
    summary_text = (
        f'$r^2$ = {r_squared:.3f}\n'
        f'RMSD = {rmsd:.2f} kJ/mol\n'
        f'MAD = {mad_kjmol:.2f} kJ/mol\n'
        f'MSD = {msd_kjmol:.2f} kJ/mol\n'
        f'Max = {max_err_val:.2f} kJ/mol'
    )
    ax.text(0.98, 0.02, summary_text, transform=ax.transAxes, horizontalalignment='right',
           verticalalignment='bottom', fontsize=fontsize_text,
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='black', linewidth=bbox_linewidth))

    # Annotate the maximum-error point (keep lightweight for small plots)
    dx = 8 if size_scale >= 0.6 else 4
    dy = 8 if size_scale >= 0.6 else 4
    ax.annotate(
        f'Max |Δ| = {max_err_val:.1f}',
        xy=(max_err_x, max_err_y),
        xytext=(dx, dy),
        textcoords='offset points',
        fontsize=max(5, int(6 * size_scale)),
        color='#d62728',
        ha='left',
        va='bottom',
        arrowprops=dict(arrowstyle='-', color='#d62728', linewidth=0.7 * size_scale, alpha=0.8),
        zorder=7,
    )
    
    # Adjust label padding based on figure size (tighter than matplotlib default)
    labelpad = max(1, int(3 * size_scale)) if size_scale < 0.6 else 3
    ax.set_xlabel(xlabel, fontsize=fontsize_label, labelpad=labelpad)
    ax.set_ylabel(ylabel, fontsize=fontsize_label, labelpad=labelpad)
    ax.legend(fontsize=fontsize_legend, frameon=True, fancybox=False, edgecolor='black', loc='upper left')
    # Reduce tick label font sizes for small plots
    ax.tick_params(labelsize=max(5, int(7 * size_scale)))
    ax.grid(True, alpha=0.3, linewidth=0.5 * size_scale)
    ax.set_aspect('equal', adjustable='box')
    
    # Use provided limits or calculate from data
    if xlim is None or ylim is None:
        # Calculate limits from data
        data_min = min(min(x), min(y))
        data_max = max(max(x), max(y))
        max_range = max(data_max - data_min, 1.0)  # Ensure at least 1.0 range
        center = (data_min + data_max) / 2
        if xlim is None:
            xlim = (center - max_range/2, center + max_range/2)
        if ylim is None:
            ylim = (center - max_range/2, center + max_range/2)
    
    # Ensure x and y have the same range for equal aspect
    x_range = xlim[1] - xlim[0]
    y_range = ylim[1] - ylim[0]
    max_range = max(x_range, y_range)
    x_center = (xlim[0] + xlim[1]) / 2
    y_center = (ylim[0] + ylim[1]) / 2
    ax.set_xlim(x_center - max_range/2, x_center + max_range/2)
    ax.set_ylim(y_center - max_range/2, y_center + max_range/2)
    
    # Reduce padding more aggressively for small plots
    if size_scale < 0.6:  # Small plots
        # Reduce margins around plot area - use subplots_adjust instead of tight_layout
        plt.subplots_adjust(left=0.18, bottom=0.18, right=0.98, top=0.98)
        pad_inches = 0.02
    else:
        plt.tight_layout()
        pad_inches = 0.1
    format_axis_offsets(ax, fontsize=max(5, int(8 * size_scale)))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=pad_inches)
    print(f"Saved: {output_path}")
    plt.close()


