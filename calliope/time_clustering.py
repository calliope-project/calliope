import matplotlib.pyplot as plt

import matplotlib.patches
from matplotlib.colors import rgb2hex
import seaborn as sns

from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import dendrogram
from scipy.spatial.distance import pdist


def fancy_dendrogram(*args, **kwargs):
    max_d = kwargs.pop('max_d', None)
    if max_d and 'color_threshold' not in kwargs:
        kwargs['color_threshold'] = max_d
    annotate_above = kwargs.pop('annotate_above', 0)

    ddata = dendrogram(*args, **kwargs)

    if not kwargs.get('no_plot', False):
        plt.xlabel('Sample index or (cluster size)')
        plt.ylabel('Distance')
        for i, d, c in zip(ddata['icoord'], ddata['dcoord'], ddata['color_list']):
            x = 0.5 * sum(i[1:3])
            y = d[1]
            if y > annotate_above:
                plt.plot(x, y, 'o', c=c)
                plt.annotate("%.3g" % y, (x, y), xytext=(0, -5),
                             textcoords='offset points',
                             va='top', ha='center', fontsize=10)
        if max_d:
            plt.axhline(y=max_d, c='grey', lw=1.0, alpha=0.5)
    return ddata


def draw_dendrogram(Z, max_d):
    fig = plt.figure(figsize=(12, 6))
    ax = plt.subplot(111)

    # link_color_pal = sns.color_palette("hls", 8)
    link_color_pal = sns.color_palette("Set2", 10)

    hierarchy.set_link_color_palette([rgb2hex(i) for i in link_color_pal])

    color_threshold = max_d

    fancy_dendrogram(
        Z,
        ax=ax,
        color_threshold=color_threshold,
        truncate_mode='lastp',
        p=20,
        leaf_rotation=90,
        leaf_font_size=10,
        show_contracted=True,
        annotate_above=5,  # useful in small plots so annotations don't overlap
        max_d=color_threshold,
    )

    sns.despine()

    # Modify the contracted markers
    for child in ax.get_children():
        if isinstance(child, matplotlib.patches.Ellipse):
            child.set_zorder(1000)
            child.set_alpha(0.3)

    return fig


def cophenetic_corr(X, Z):
    """
    Get the Cophenetic Correlation Coefficient of a clustering with help
    of the cophenet() function. This (very very briefly) compares (correlates)
    the actual pairwise distances of all your samples to those implied by the
    hierarchical clustering. The closer the value is to 1, the better
    the clustering preserves the original distances.

    Source:
    https://joernhees.de/blog/2015/08/26/scipy-hierarchical-clustering-and-dendrogram-tutorial/

    """
    c, coph_dists = hierarchy.cophenet(Z, pdist(X))
    return c, coph_dists
