import pandas as pd
import matplotlib

matplotlib.use('tkAgg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import numpy as np

# %%
df = pd.read_csv("Transformed Data Set - Sheet1.csv")

X_raw = df.drop("Gender", axis=1)
y_raw = df["Gender"]

encoder = OneHotEncoder(sparse_output=False)
X = encoder.fit_transform(X_raw)

label_encoder = LabelEncoder()
y_int = label_encoder.fit_transform(y_raw)

Y = np.eye(len(np.unique(y_int)))[y_int]


# %%


def extend_input_with_bias(network_input):
    bias_extension = np.ones(network_input.shape[1]).reshape(1, -1)
    network_input = np.vstack([bias_extension, network_input])
    return network_input


# %%


X = X.T
Y = Y.T


# %%


def create_network(input_size, output_size, hidden_sizes):
    network = []
    layer_sizes = hidden_sizes
    layer_sizes.append(output_size)
    for neuron_count in layer_sizes:
        layer = np.random.rand(input_size + 1, neuron_count) * 2 - 1
        input_size = neuron_count
        network.append(layer)
    return network


# %%


def unipolar_activation(u):
    return 1 / (1 + np.exp(-u))


def unipolar_derivative(u):
    a = unipolar_activation(u)
    return a * (1.0 - a)


# %%


def feed_forward(network_input, network):
    layer_input = network_input
    responses = []

    for weights in network:
        layer_input = extend_input_with_bias(layer_input)
        response = unipolar_activation(weights.T @ layer_input)
        layer_input = response
        responses.append(response)

    return responses


def predict(network_input, network):
    return feed_forward(network_input, network)[-1]


def calculate_mse(predicted, expected):
    return np.sum((predicted - expected) ** 2) / len(predicted)


# %%


def backpropagate(network, responses, expected_output_layer_response):
    gradients = []
    error = responses[-1] - expected_output_layer_response
    for weights, response in zip(reversed(network), reversed(responses)):
        gradient = error + unipolar_derivative(response)
        gradients.append(gradient)
        error = weights @ gradient
        error = error[1:, :]
    return list(reversed(gradients))


# %%


def clculate_weights_changes(network, network_input, network_responses, gradients, learning_factor):
    layer_inputs = [network_input] + network_responses[:-1]
    weights_changes = []
    for weights, layer_input, gradient in zip(network, layer_inputs, gradients):
        layer_input = extend_input_with_bias(layer_input)
        change = layer_input.dot(gradient.T) * learning_factor
        weights_changes.append(change)
    return weights_changes


# %%


def adjust_weights(network, changes):
    new_network = []
    for weights, change in zip(network, changes):
        new_weights = weights - change
        new_network.append(new_weights)
    return new_network


# %%


def train_network(network, network_input, expected_output, learning_factor, epochs):
    mse_history = []
    for _ in range(epochs):
        responses = feed_forward(network_input, network)
        mse_history.append(calculate_mse(responses[-1], expected_output))
        gradients = backpropagate(network, responses, Y)
        changes = clculate_weights_changes(network, network_input, responses, gradients, learning_factor)
        network = adjust_weights(network, changes)
    mse_history.append(calculate_mse(responses[-1], expected_output))
    return network, np.asarray(mse_history)


# %%


network = create_network(X.shape[0], Y.shape[0], [12, 8])
network, mse_history = train_network(network, X, Y, 0.001, 20)

plt.plot(mse_history)
plt.show()
