from typing import Union, List

import numpy as np
from tensorflow.keras import Model
from tensorflow.keras.layers import BatchNormalization, Dense
from tensorflow.keras.regularizers import l1, l2, l1_l2
from tensorflow.keras import Input

from neural_network_preprocessing.importance import ImportanceType


def modify_model(model: Model, class_index: int, importance_type: ImportanceType) -> Model:
    gamma_initializer: str = "zeros"
    if importance_type & ImportanceType.GAMMA:
        gamma_initializer = "ones"

    gamma_regularizer = None
    if importance_type & ImportanceType.L1 and not importance_type & ImportanceType.L2:
        gamma_regularizer = l1()
    if not importance_type & ImportanceType.L1 and importance_type & ImportanceType.L2:
        gamma_regularizer = l2()
    if importance_type & ImportanceType.L1 and importance_type & ImportanceType.L2:
        gamma_regularizer = l1_l2()

    max_layer: int = len(model.layers)
    last_output: Input = None
    network_input: Input = None
    for i, layer in enumerate(model.layers):
        if i == 0:
            last_output = layer.output
            network_input = layer.input
        if 0 < i < max_layer:
            new_layer: Union[BatchNormalization, BatchNormalization] = BatchNormalization(
                center=(importance_type & ImportanceType.CENTERING),
                gamma_initializer=gamma_initializer,
                gamma_regularizer=gamma_regularizer)
            last_output = new_layer(last_output)
        if i == max_layer - 1:
            new_end_layer: Dense = Dense(2, activation="softmax", name="binary_output_layer")
            last_output = new_end_layer(last_output)

            old_weights = layer.get_weights()
            old_weights[0] = np.transpose(old_weights[0], (1, 0))
            new_weights: List[np.array] = [
                np.append(old_weights[0][class_index:class_index + 1],
                          np.subtract(np.sum(old_weights[0], axis=0, keepdims=True),
                                      old_weights[0][class_index:class_index + 1]), axis=0),
                np.append(old_weights[1][class_index:class_index + 1],
                          np.subtract(np.sum(old_weights[1], axis=0, keepdims=True),
                                      old_weights[1][class_index:class_index + 1]), axis=0)
            ]
            new_weights[0] = np.transpose(new_weights[0], (1, 0))
            new_end_layer.set_weights(new_weights)
        elif i > 0:
            last_output = layer(last_output)

    return Model(inputs=network_input, outputs=last_output)
