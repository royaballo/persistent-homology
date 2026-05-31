"""Plotting utilities for persistence diagrams and benchmark results.

The diagram plotter is generic: all four synthetic datasets share the
same persistence-diagram layout, so a single function draws the diagram
and an optional list of annotations highlights dataset-specific features
(the loop of a circle, the void of a sphere, and so on).
"""

import matplotlib.patches as mpatches  # type: ignore[import-untyped]
import matplotlib.pyplot as plt  # type: ignore[import-untyped]

# Graph parameters.
plt.rcParams.update(
    {
        "font.family": "serif",
        "axes.labelsize": 14,
        "font.size": 13,
        "legend.fontsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "figure.titlesize": 16,
    }
)

DIM_COLOR = {0: "#1565C0", 1: "#C62828", 2: "#2E7D32"}
DIM_LABEL = {0: r"$H_0$", 1: r"$H_1$", 2: r"$H_2$"}


# Diagram helpers
def proxy_infinity(diagram, margin=1.12):
    """Numerical stand-in for "+inf": the largest finite death, scaled.

    Essential features have an infinite death and cannot be plotted at
    their true ordinate; they are drawn at this proxy height instead.
    """
    finites = [
        d for intervals in diagram.values() for (_, d) in intervals if d != float("inf")
    ]
    return max(finites) * margin if finites else 1.0


def ripser_to_diagram(result):
    """Convert Ripser's "dgms" output into our diagram dictionary format."""
    diagram = {}
    for dim, dgm in enumerate(result["dgms"]):
        diagram[dim] = [(float(b), float(d)) for (b, d) in dgm]
    return diagram


# Persistence diagram


def plot_persistence_diagram(
    diagram, max_dim=2, annotations=None, title=None, ax=None, savepath=None
):
    """Draw a persistence diagram.

    Parameters
    ----------
    diagram : dict
        Maps each dimension to a list of "(birth, death)" pairs.
    max_dim : int
        Highest homology dimension to display.
    annotations : list of dict, optional
        Each dict describes an arrow annotation with keys "xy" (the
        point), "text" (the label), "dim" (which dimension's colour
        to use), and optionally "xytext" (the label position).
    title : str, optional
        Axis title.
    ax : matplotlib axis, optional
        Axis to draw on; a new figure is created if omitted.
    savepath : str, optional
        If given, the figure is saved there at 150 dpi.

    Returns
    -------
    ax : matplotlib axis
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))

    proxy = proxy_infinity(diagram)
    lim = proxy * 1.12

    # Diagonal and the infinity line.
    ax.plot([0, lim], [0, lim], "--", color="#9E9E9E", lw=1.0, zorder=1)
    ax.axhline(proxy, color="#9E9E9E", lw=0.8, ls=":", alpha=0.6)
    ax.text(
        lim * 0.01, proxy * 1.01, r"$\infty$", color="#9E9E9E", fontsize=13, va="bottom"
    )

    # Scatter points, one colour per dimension. Finite features are drawn
    # as circles; essential features as triangles at the proxy height.
    legend_handles = []
    for dim in range(max_dim + 1):
        intervals = diagram.get(dim, [])
        if not intervals:
            continue
        color = DIM_COLOR[dim]
        births_f, deaths_f, births_i = [], [], []
        for b, d in intervals:
            if d == float("inf"):
                births_i.append(b)
            else:
                births_f.append(b)
                deaths_f.append(d)
        if births_f:
            ax.scatter(
                births_f,
                deaths_f,
                c=color,
                s=60,
                marker="o",
                zorder=4,
                linewidths=0,
                alpha=0.85,
            )
        if births_i:
            ax.scatter(
                births_i,
                [proxy] * len(births_i),
                c=color,
                s=80,
                marker="^",
                zorder=4,
                linewidths=0,
                alpha=0.90,
            )
        legend_handles.append(mpatches.Patch(color=color, label=DIM_LABEL[dim]))

    # Optional dataset-specific annotations.
    for ann in annotations or []:
        dim = ann.get("dim", 1)
        color = DIM_COLOR[dim]
        xy = ann["xy"]
        xytext = ann.get("xytext", (xy[0] + lim * 0.18, xy[1] - lim * 0.15))
        ax.annotate(
            ann["text"],
            xy=xy,
            xytext=xytext,
            fontsize=11,
            color=color,
            arrowprops=dict(arrowstyle="->", color=color, lw=1.0, mutation_scale=10),
            bbox=dict(
                boxstyle="round,pad=0.25", fc="white", ec=color, alpha=0.85, lw=0.9
            ),
        )

    ax.legend(
        handles=legend_handles, loc="lower right", framealpha=0.85, edgecolor="#E0E0E0"
    )
    ax.set_xlabel("Birth")
    ax.set_ylabel("Death")
    ax.set_xlim(-lim * 0.04, lim)
    ax.set_ylim(-lim * 0.04, lim * 1.08)
    ax.set_aspect("equal")
    ax.grid(color="#F0F0F0", lw=0.7, zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    if title:
        ax.set_title(title)

    if savepath:
        ax.figure.savefig(savepath, dpi=150, bbox_inches="tight")
        print(f"Saved -> {savepath}")
    return ax


# Benchmark plots

# Fixed visual style per method: our three implementations in blues,
# GUDHI in orange, Ripser in green.
METHOD_STYLE = {
    "our_standard": {
        "label": "Standard",
        "color": "#1565C0",
        "marker": "o",
        "linestyle": "-",
    },
    "our_clearing": {
        "label": "Clearing",
        "color": "#42A5F5",
        "marker": "s",
        "linestyle": "--",
    },
    "our_cohomology": {
        "label": "Cohomology",
        "color": "#0D47A1",
        "marker": "^",
        "linestyle": ":",
    },
    "GUDHI": {"label": "GUDHI", "color": "#E65100", "marker": "D", "linestyle": "-"},
    "Ripser": {"label": "Ripser", "color": "#2E7D32", "marker": "*", "linestyle": "-"},
}

METHOD_ORDER = ["our_standard", "our_clearing", "our_cohomology", "GUDHI", "Ripser"]


# Per-quantity y-axis configuration.
_QUANTITY_CONFIG = {
    "time": {
        "default_y_col": "time_s",
        "ylabel": "Computation time (seconds)",
        "y_ticks": [0.001, 0.01, 0.1, 1, 10, 100],
        "y_tick_labels": ["1 ms", "10 ms", "0.1 s", "1 s", "10 s", "100 s"],
    },
    "memory": {
        "default_y_col": "peak_mb",
        "ylabel": "Peak memory (MB)",
        "y_ticks": [1, 10, 100, 1000],
        "y_tick_labels": ["1 MB", "10 MB", "100 MB", "1 GB"],
    },
}

# Per-x-axis configuration. The default axis is the number of points;
# the 'max_eps' axis is used for the filtration-scope experiment, where
# the x variable is the maximum filtration value instead.
_XAXIS_CONFIG = {
    "n": {
        "default_col": "n",
        "xlabel": "Number of points $n$",
        "exponent_var": "n",
    },
    "max_eps": {
        "default_col": "max_eps",
        "xlabel": r"Maximum filtration value $\varepsilon_{\max}$",
        "exponent_var": None,
    },
}


def plot_loglog_scaling(
    df,
    quantity="time",
    xaxis="n",
    exponents=None,
    x_col=None,
    y_col=None,
    method_col="method",
    title=None,
    savepath=None,
):
    """Log-log plot of a scaling quantity against a chosen x-axis.

    Supports two y-axes ("quantity='time'" or "'memory'") and two
    x-axes ("xaxis='n'" for size-scaling experiments,
    "xaxis='max_eps'" for filtration-scope experiments). The empirical
    scaling exponents, when given, are displayed in the legend only.
    Exponents are only meaningful when plotting against "n"; they are
    silently ignored for "xaxis='max_eps'".

    Parameters
    ----------
    df : pandas DataFrame
        Must contain the x, y, and method columns named below.
    quantity : {'time', 'memory'}
        Selects the y-axis label and tick formatting. The default y-axis
        column name is also inferred from this.
    xaxis : {'n', 'max_eps'}
        Selects the x-axis label and the default x-axis column name.
    exponents : dict, optional
        Maps a method name to its fitted scaling exponent. When given,
        each legend entry is annotated with "n^p". Only used when
        "xaxis='n'".
    x_col : str, optional
        Column holding the x values; defaults to the natural column
        of the chosen x-axis ("'n'" or "'max_eps'").
    y_col : str, optional
        Column holding the values; defaults to the quantity's natural
        column ("'time_s'" or "'peak_mb'").
    method_col : str
    title : str, optional
    savepath : str, optional

    Returns
    -------
    ax : matplotlib axis
    """
    if quantity not in _QUANTITY_CONFIG:
        raise ValueError(
            f"unknown quantity {quantity!r}; "
            f"expected one of {list(_QUANTITY_CONFIG)}"
        )
    if xaxis not in _XAXIS_CONFIG:
        raise ValueError(
            f"unknown xaxis {xaxis!r}; " f"expected one of {list(_XAXIS_CONFIG)}"
        )
    ycfg = _QUANTITY_CONFIG[quantity]
    xcfg = _XAXIS_CONFIG[xaxis]
    if y_col is None:
        y_col = ycfg["default_y_col"]
    if x_col is None:
        x_col = xcfg["default_col"]

    _, ax = plt.subplots(figsize=(9, 6))

    # Plot methods in a fixed order, so the legend is always consistent.
    methods_present = [m for m in METHOD_ORDER if m in df[method_col].unique()]
    for method in methods_present:
        group = df[df[method_col] == method].sort_values(x_col)
        # Drop rows where the measurement is missing (e.g. MemoryError).
        group = group[group[y_col].notna()]
        if group.empty:
            continue
        style = METHOD_STYLE[method]

        label = style["label"]
        # Exponent annotations only make sense for size scaling.
        if exponents and method in exponents and xcfg["exponent_var"]:
            entry = exponents[method]
            if isinstance(entry, tuple):
                slope, stderr = entry
                if stderr == stderr:  # not NaN
                    label = (
                        f"{label}  (${xcfg['exponent_var']}"
                        f"^{{{slope:.2f} \\pm {stderr:.2f}}}$)"
                    )
                else:
                    label = f"{label}  (${xcfg['exponent_var']}" f"^{{{slope:.2f}}}$)"
            else:
                label = f"{label}  (${xcfg['exponent_var']}" f"^{{{entry:.2f}}}$)"

        ax.plot(
            group[x_col],
            group[y_col],
            label=label,
            color=style["color"],
            marker=style["marker"],
            markersize=8,
            linestyle=style["linestyle"],
            linewidth=1.8,
            markeredgecolor="white",
            markeredgewidth=0.5,
            alpha=0.95,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(xcfg["xlabel"])
    ax.set_ylabel(ycfg["ylabel"])

    # x-ticks at the actual measurement values, not powers of ten.
    x_values = sorted(df[x_col].unique())
    ax.set_xticks(x_values)
    # Integer formatting for n, decimal formatting for max_eps.
    if xaxis == "n":
        ax.set_xticklabels([str(int(x)) for x in x_values])
    else:
        ax.set_xticklabels([f"{x:g}" for x in x_values])
    ax.minorticks_off()

    # Human-readable y-tick labels for the chosen quantity.
    ax.set_yticks(ycfg["y_ticks"])
    ax.set_yticklabels(ycfg["y_tick_labels"])

    ax.grid(True, which="major", color="#E0E0E0", lw=0.7, zorder=0)
    ax.legend(loc="lower right", framealpha=0.9, edgecolor="#E0E0E0", handlelength=2.4)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    if title:
        ax.set_title(title)

    if savepath:
        ax.figure.savefig(savepath, dpi=160, bbox_inches="tight")
        print(f"Saved -> {savepath}")
    return ax
