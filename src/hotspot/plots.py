import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import Colormap, ListedColormap
from scipy.cluster.hierarchy import leaves_list

from .types import FloatArray


def local_correlation_plot(
    local_correlation_z: pd.DataFrame,
    modules: pd.Series,
    linkage: FloatArray,
    mod_cmap: str | Colormap = "tab10",
    vmin: float = -8,
    vmax: float = 8,
    z_cmap: str | Colormap = "RdBu_r",
    yticklabels: bool = False,
) -> None:
    """Clustered heatmap of gene-gene local correlation Z-scores, annotated by module."""
    cmap = plt.get_cmap(mod_cmap)
    if not isinstance(cmap, ListedColormap):
        raise TypeError("`mod_cmap` must be a discrete (listed) colormap, e.g. 'tab10'")
    colors = [cmap(i) for i in range(cmap.N)]
    module_colors = {i: colors[(i - 1) % len(colors)] for i in modules.unique()}
    module_colors[-1] = "#ffffff"

    row_colors1 = pd.Series(
        [module_colors[i] for i in modules],
        index=local_correlation_z.index,
    )

    row_colors = pd.DataFrame(
        {
            "Modules": row_colors1,
        }
    )

    cm = sns.clustermap(
        local_correlation_z,
        row_linkage=linkage,
        col_linkage=linkage,
        vmin=vmin,
        vmax=vmax,
        cmap=z_cmap,
        xticklabels=False,
        yticklabels=yticklabels,
        row_colors=row_colors,
        rasterized=True,
    )

    plt.sca(cm.ax_heatmap)
    plt.ylabel("")
    plt.xlabel("")

    cm.ax_row_dendrogram.remove()

    # Add 'module X' annotations
    ii = leaves_list(linkage)

    mod_reordered = modules.iloc[ii]

    mod_map = {}
    y = np.arange(modules.size)

    for x in mod_reordered.unique():
        if x == -1:
            continue

        mod_map[x] = y[mod_reordered == x].mean()

    if cm.ax_row_colors is None:  # pragma: no cover
        raise ValueError("clustermap did not create row-color axes.")
    plt.sca(cm.ax_row_colors)
    for mod, mod_y in mod_map.items():
        plt.text(-0.5, y=mod_y, s=f"Module {mod}", horizontalalignment="right", verticalalignment="center")
    plt.xticks([])

    if cm.ax_cbar is None:  # pragma: no cover
        raise ValueError("clustermap did not create a colorbar axes.")
    cm.ax_cbar.set_ylabel("Z-Scores")
    cm.ax_cbar.yaxis.set_label_position("left")
