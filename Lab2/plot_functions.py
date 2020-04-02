import numpy as np

import matplotlib
import matplotlib.pyplot as plt

from utils import *


def _add_matrix_to_heatmap(im, matrix):
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if np.isnan(matrix[i,j]):
                continue

            im.axes.text(j, i, round(matrix[i,j], 3), 
                         horizontalalignment='center',
                         verticalalignment='center',
                         color='w')
    return im


def plot_weight_matrix(obj, ax=None, show_values=True):
    layers = get_layers(obj)
    matrix = get_weight_matrix(layers)

    x_labels = name_units(layers)
    y_labels = name_units(layers, inputs=True)

    x_indices = np.cumsum(layer_sizes(layers[:-1]))
    y_indices = np.cumsum(layer_sizes(layers[:-1], inputs=True))

    if ax is None:
        ax = plt.gca()

    # Plot heatmap
    im = ax.imshow(matrix, cmap='bwr')
    ax.set_xlabel('target unit')
    ax.set_ylabel('source unit')
    
    # Separate layers by white lines
    ax.grid(which='minor', color='k', linestyle='-', linewidth=2)
    ax.set_xticks(x_indices - 0.5, minor=True)
    ax.set_yticks(y_indices - 0.5, minor=True)

    # Show all node names as ticks
    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_yticks(np.arange(len(y_labels)))

    ax.set_xticklabels(x_labels)
    ax.set_yticklabels(y_labels)

    # Tick positioning
    ax.xaxis.tick_top()
    plt.xticks(rotation=90)

    # Show colorbar
    cbar = ax.figure.colorbar(im)
    cbar.ax.set_ylabel('weight', rotation=-90, va='bottom')

    # Add weight as text
    if show_values:
        _add_matrix_to_heatmap(im, matrix)


def plot_activation_matrix(model, X, y=None, subset=None, layers=None, ax=None, 
                           in_labels=None, out_labels=None, show_values=True, 
                           vmin=None, vmax=None):
    matrix = get_activation_matrix(model, X, layers)

    if subset is not None:
        matrix = matrix[subset]
        X = X[subset]
        if y is not None:
            y = y[subset]

    if layers is None:
        layers = model.layers

    x_indices = np.cumsum([layer.units for layer in layers[:-1]])
    
    if out_labels is None:
        out_labels = name_units(layers)
    if in_labels is None:
        in_labels = X
        if y is not None:
            in_labels = [str(x_val) + ' '+ str(y_val) for x_val, y_val in zip(X, y)]

    if ax is None:
        ax = plt.gca()

    # Plot heatmap
    im = ax.imshow(matrix, cmap='bwr', vmin=vmin, vmax=vmax)
    ax.set_xlabel('activated unit')
    ax.set_ylabel('input unit')
    
    # Separate layers and inputs by white lines
    ax.grid(which='minor', color='k', linestyle='-', linewidth=2)
    ax.set_xticks(x_indices - 0.5, minor=True)
    ax.set_yticks(np.arange(1, matrix.shape[0]) - 0.5, minor=True)

    # Show all node names as ticks
    ax.set_xticks(np.arange(len(out_labels)))
    ax.set_yticks(np.arange(X.shape[0]))

    ax.set_xticklabels(out_labels)
    ax.set_yticklabels(in_labels)

    # Tick positioning
    ax.xaxis.tick_top()
    plt.xticks(rotation=90)

    # Show colorbar
    cbar = ax.figure.colorbar(im)
    cbar.ax.set_ylabel('activation', rotation=-90, va='bottom')

    # Add weight as text
    if show_values:
        _add_matrix_to_heatmap(im, matrix)
