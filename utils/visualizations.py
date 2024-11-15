from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import cm
from umap import UMAP
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE


_default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
_my_colors = np.array([
    [0, 127, 70],
    [255, 0, 0],
    [255, 217, 38],
    [0, 135, 255],
    [165, 0, 165],
    [255, 167, 255],
    [97, 142, 151],
    [0, 255, 255],
    [255, 96, 38],
    [142, 76, 0],
    [33, 0, 127],
    [0, 0, 0],
    [183, 183, 183],
    [76, 255, 0],
], dtype=float) / 255


def generate_colors(n):
    cm = plt.get_cmap('gist_rainbow')
    colors = [cm(1.*i/n) for i in range(n)]
    return colors

def plot_similarity_matrix(matrix, labels_a=None, labels_b=None, ax: plt.Axes=None, title=""):
    if ax is None:
        _, ax = plt.subplots()
    fig = plt.gcf()
        
    img = ax.matshow(matrix, extent=(-0.5, matrix.shape[0] - 0.5, 
                                     -0.5, matrix.shape[1] - 0.5))

    ax.xaxis.set_ticks_position("bottom")
    if labels_a is not None:
        ax.set_xticks(range(len(labels_a)))
        ax.set_xticklabels(labels_a, rotation=90)
    if labels_b is not None:
        ax.set_yticks(range(len(labels_b)))
        ax.set_yticklabels(labels_b[::-1])  # Upper origin -> reverse y axis
    ax.set_title(title)

    cax = make_axes_locatable(ax).append_axes("right", size="5%", pad=0.15)
    fig.colorbar(img, cax=cax, ticks=np.linspace(0.4, 1, 7))
    img.set_clim(0.4, 1)
    img.set_cmap("inferno")
    
    return ax
    
    
def plot_histograms(all_samples, ax=None, names=None, title=""):
    """
    Plots (possibly) overlapping histograms and their median 
    """
    if ax is None:
        _, ax = plt.subplots()
    
    for samples, color, name in zip(all_samples, _default_colors, names):
        ax.hist(samples, density=True, color=color + "80", label=name)
    ax.legend()
    ax.set_xlim(0.35, 1)
    ax.set_yticks([])
    ax.set_title(title)
        
    ylim = ax.get_ylim()
    ax.set_ylim(*ylim)      # Yeah, I know
    for samples, color in zip(all_samples, _default_colors):
        median = np.median(samples)
        ax.vlines(median, *ylim, color, "dashed")
        ax.text(median, ylim[1] * 0.15, "median", rotation=270, color=color)
    
    return ax


def plot_projections(embeds, speakers, ax=None, colors=None, markers=None, legend=True, 
                     title="", cluster_name="", labels=None, **kwargs):
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))
        
    if cluster_name == 'spectral':
        reducer = TSNE(init='pca', **kwargs)
    if cluster_name == 'umap_hdbscan':
        reducer = UMAP(**kwargs)
    
    # Compute the 2D projections. You could also project to another number of dimensions (e.g. 
    # for a 3D plot) or use a different different dimensionality reduction like PCA or TSNE.
    
    projs = reducer.fit_transform(embeds)
    
    # Draw the projections
    speakers = np.array(speakers)
    colors = generate_colors(len(np.unique(speakers)))
    colors = colors or _my_colors
    for i, speaker in enumerate(np.unique(speakers)):
        speaker_projs = projs[speakers == speaker]
        marker = "o" if markers is None else markers[i]
        label = speaker if legend else None
        ax.scatter(*speaker_projs.T, s=60, c=[colors[i]], marker=marker, label=label, edgecolors='k')
        if labels is not None:
            for j, (proj_x, proj_y) in enumerate(speaker_projs):
                label_index = np.where(speakers == speaker)[0][j]
                ax.text(proj_x, proj_y, str(labels[label_index]), fontsize=8, ha='right')
        center = speaker_projs.mean(axis=0)
        ax.scatter(*center, s=200, c=[colors[i]], marker="X", edgecolors='k')

        
    if legend:
        ax.legend(title="Speakers", ncol=2)
    ax.set_title(title)
    #ax.set_xticks([])
    #ax.set_yticks([])
    ax.grid(True)
    ax.set_aspect("equal")
    
    return projs

def plot_embedding_as_heatmap(embed, ax=None, title="", shape=None, color_range=(0, 0.30)):
    if ax is None:
        _, ax = plt.subplots()
    
    if shape is None:
        height = int(np.sqrt(len(embed)))
        shape = (height, -1)
    embed = embed.reshape(shape)
    
    cmap = cm.get_cmap()
    mappable = ax.imshow(embed, cmap=cmap)
    cbar = plt.colorbar(mappable, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_clim(*color_range)
    
    ax.set_xticks([]), ax.set_yticks([])
    ax.set_title(title)