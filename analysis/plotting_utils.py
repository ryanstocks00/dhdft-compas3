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
    """Calculate R², RMSD, and MAD statistics."""
    correlation = np.corrcoef(x, y)[0, 1]
    r_squared = correlation ** 2
    rmsd = np.sqrt(np.mean((x - y) ** 2))
    abs_errors, abs_x = np.abs(x - y), np.abs(x)
    mad_percentage = np.mean(np.where(abs_x > 0, abs_errors / abs_x, 0)) * 100
    return r_squared, rmsd, mad_percentage


def format_axis_offsets(ax):
    """Move axis offset text to corners."""
    for axis, pos, ha, va in [(ax.xaxis, (0.98, 0.98), 'right', 'top'), (ax.yaxis, (0.02, 0.02), 'left', 'bottom')]:
        offset = axis.get_offset_text()
        if offset.get_text():
            offset.set_visible(False)
            ax.text(pos[0], pos[1], offset.get_text(), transform=ax.transAxes,
                   horizontalalignment=ha, verticalalignment=va, fontsize=8, color=offset.get_color())


def create_scatter_plot(x, y, xlabel, ylabel, output_path, mad_kjmol=None):
    """Create a standardized scatter plot with statistics.
    
    Args:
        x: x-axis data
        y: y-axis data
        xlabel: x-axis label
        ylabel: y-axis label
        output_path: path to save the plot
        mad_kjmol: Mean Absolute Deviation in kJ/mol (optional, will be calculated if not provided)
    """
    from pathlib import Path
    
    fig, ax = plt.subplots(figsize=(SINGLE_COLUMN_WIDTH, SINGLE_COLUMN_WIDTH))
    ax.scatter(x, y, alpha=0.3, s=6, color='#1f77b4', edgecolors='none', linewidth=0)
    
    min_val, max_val = min(min(x), min(y)), max(max(x), max(y))
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=1.0, label='Perfect agreement')
    
    if len(x) > 1:
        slope, intercept = np.polyfit(x, y, 1)
        trendline_x = np.array([min_val, max_val])
        ax.plot(trendline_x, slope * trendline_x + intercept, 'black', alpha=0.8, linewidth=0.9,
               linestyle='-', label='Linear fit', zorder=10)
    
    r_squared, rmsd, mad_percentage = calculate_stats(x, y)
    if mad_kjmol is None:
        mad_kjmol = np.mean(np.abs(x - y))
    
    summary_text = f'$R^2$ = {r_squared:.3f}\nRMSD = {rmsd:.2f} kJ/mol\nMAD = {mad_kjmol:.2f} kJ/mol\nMAD = {mad_percentage:.2f}%'
    ax.text(0.98, 0.02, summary_text, transform=ax.transAxes, horizontalalignment='right',
           verticalalignment='bottom', fontsize=7,
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='black', linewidth=0.5))
    
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.legend(fontsize=7, frameon=True, fancybox=False, edgecolor='black', loc='upper left')
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.set_aspect('equal', adjustable='box')
    
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    max_range = max(xlim[1] - xlim[0], ylim[1] - ylim[0])
    xcenter, ycenter = (xlim[0] + xlim[1]) / 2, (ylim[0] + ylim[1]) / 2
    ax.set_xlim(xcenter - max_range/2, xcenter + max_range/2)
    ax.set_ylim(ycenter - max_range/2, ycenter + max_range/2)
    
    plt.tight_layout()
    format_axis_offsets(ax)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


