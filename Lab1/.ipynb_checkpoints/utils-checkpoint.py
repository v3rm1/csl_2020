import numpy as np

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Wrapper, RNN
from tensorflow.keras import backend as K


def get_layers(obj):
    """
    Accepts a Keras Model, a Layer or a list of Layers and returns 
    the list of Layers.
    """
    
    if isinstance(obj, Model):
        obj = obj.layers
    
    if isinstance(obj, Wrapper):
        return [obj.layer]
    
    if isinstance(obj, Layer):
        return [obj]
    
    if isinstance(obj, list):
        return [get_layers(layer)[0] for layer in obj]

    return [None]


def name_units(obj, inputs=False, fmt='{} #{}'):
    """ 
    Names all units in a list of Layers based on Layer name and 
    unit index.
    """
    
    layers = get_layers(obj)
    units = []
    
    if inputs:
        input_dim = layers[0].input_shape[-1]
        for i in range(input_dim):
            units.append(fmt.format('input', i))

    for i, layer in enumerate(layers, 1):
        if isinstance(layer, RNN) and inputs:
            for u in range(layer.units):
                units.append(fmt.format(layer.name + ' state', u))

        if not inputs or i < len(layers):
            for i in range(layer.units):
                units.append(fmt.format(layer.name, i))
    
    if inputs:
        units.append('bias')

    return units


def layer_sizes(obj, inputs=False):
    layers = get_layers(obj)
    
    sizes = []
    if inputs and len(layers) > 0:
        sizes.append(layers[0].input_shape[-1])
    
    for layer in layers:
        if isinstance(layer, RNN) and inputs:
            sizes.append(layer.units)
        sizes.append(layer.units)

    return sizes


def get_weight_matrix(obj, include_bias=True):
    """
    Returns a weight matrix for a Model, a Layer or a list of Layers
    """
    layers = get_layers(obj)
    ncols = sum(layer.units for layer in layers)

    weight_arrays = []
    bias_array = np.array([], dtype='float32')

    for layer in layers:
        if isinstance(layer, RNN):
            weights, state_weights, biases = layer.get_weights()
            weights = np.vstack((weights, state_weights))
        else:
            weights, biases = layer.get_weights()  # TODO Test on layers without bias

        pad_width = (len(bias_array), ncols - len(bias_array) - layer.units)
        weights = np.apply_along_axis(np.pad, 1, weights, pad_width, 'constant', constant_values=np.nan)

        weight_arrays.append(weights)
        bias_array = np.append(bias_array, biases)

    if include_bias:
        weight_arrays = weight_arrays + [bias_array]

    return np.vstack(weight_arrays)


def get_activation_matrix(model, X, layers=None):
    """
    Returns the activation matrix for a Model, a Layer or a list of Layers
    and a set of Model input values.
    """

    if layers is None:
        layers = model.layers

    targets = [layer.output for layer in layers]

    outputs = K.function([model.layers[0].input], targets)
    activations = outputs([np.array(X)])

    return np.concatenate(activations, axis=-1)