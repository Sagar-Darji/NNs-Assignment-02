# Darji, Sagar Vishnubhai
# 1002_202_201
# 2024_10_06
# Assignment_02_01

import numpy as np
import torch
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader, TensorDataset

def initialize_weights(layers, input_dim):
    weights = []
    for i in range(len(layers)):
        if i == 0:
            in_dim = input_dim
        else:
            in_dim = layers[i - 1]
        out_dim = layers[i]
        W = np.random.randn(in_dim + 1, out_dim).astype(np.float32)
        weights.append(W)
        # Print the weights for debugging
        print(f"Initialized Weight Matrix {i}:")
        print(W)
    return weights

# Define helper functions after importing torch
def forward_pass(x, weights_torch, activations):
     # Perform a forward pass through the network.
    input = x
    for idx, (W, activation) in enumerate(zip(weights_torch, activations)):
        ones = torch.ones(input.shape[0], 1)
        input_with_bias = torch.cat((ones, input), dim=1)
        z = input_with_bias @ W

        if activation.lower() == 'linear':
            a = z
        elif activation.lower() == 'sigmoid':
            a = torch.sigmoid(z)
        elif activation.lower() == 'relu':
            a = torch.relu(z)
        else:
            raise ValueError(f"Unknown activation function: {activation}")
        input = a
    return input

def compute_loss(y_pred, y_true, loss_func):
    # Compute the loss between the predictions and the true values.
    if loss_func.lower() == 'mse':
        loss = torch.mean((y_pred - y_true) ** 2)
    elif loss_func.lower() == 'ce':
        # Cross-Entropy Loss
        logits = y_pred
        exp_logits = torch.exp(logits - torch.max(logits, dim=1, keepdim=True)[0])
        probs = exp_logits / torch.sum(exp_logits, dim=1, keepdim=True)
        correct_class = torch.argmax(y_true, dim=1)
        correct_logprobs = -torch.log(probs[torch.arange(y_true.shape[0]), correct_class] + 1e-15)
        loss = torch.mean(correct_logprobs)
    elif loss_func.lower() == 'svm':
        # SVM Loss
        y_true_indices = torch.argmax(y_true, dim=1)
        batch_size = y_pred.shape[0]
        correct_class_scores = y_pred[torch.arange(batch_size), y_true_indices].view(-1, 1)
        margins = y_pred - correct_class_scores + 1.0
        margins[torch.arange(batch_size), y_true_indices] = 0
        margins = torch.clamp(margins, min=0)
        loss = torch.mean(torch.sum(margins, dim=1))
    else:
        raise ValueError(f"Unknown loss function: {loss_func}")
    return loss

def backward_pass(loss, weights_torch, alpha):
    # Perform a backward pass and update the weights.
    loss.backward()
    with torch.no_grad():
        for W in weights_torch:
            W -= alpha * W.grad
            W.grad.zero_()

def compute_validation_error(x_val, y_val, weights_torch, activations):
    # Compute the error on the validation set.
    with torch.no_grad():
        y_pred_val = forward_pass(x_val, weights_torch, activations)
        error = torch.mean(torch.abs(y_pred_val - y_val)).item()
    return error, y_pred_val.numpy()


def multi_layer_nn_torch(x_train, y_train, layers, activations, alpha=0.01, batch_size=32, epochs=0,
                         loss_func='mse', val_split=(0.8, 1.0), seed=7321):
    np.random.seed(seed)
    # Initialize weights before any torch operations
    if isinstance(layers[0], int):
        input_dim = x_train.shape[1]
        weights = initialize_weights(layers, input_dim)
        # Optional: Print weights for debugging
        print("Initialized weights:")
        print("Weight matrix 0:")
        print(weights[0])
        print("Weight matrix 1:")
        print(weights[1])
    else:
        weights = layers


    # Prepare data as NumPy arrays first
    x_train_np = np.array(x_train, dtype=np.float32)
    y_train_np = np.array(y_train, dtype=np.float32)

    total_samples = x_train_np.shape[0]
    val_start = int(np.floor(val_split[0] * total_samples))
    val_end = int(np.floor(val_split[1] * total_samples))

    x_val_np = x_train_np[val_start:val_end]
    y_val_np = y_train_np[val_start:val_end]

    x_train_np = x_train_np[:val_start]
    y_train_np = y_train_np[:val_start]

    num_batches = int(np.ceil(x_train_np.shape[0] / batch_size))
    errors = []

    # Import torch after initializing weights
    import torch

    # Convert data to torch tensors after weight initialization
    x_train_torch = torch.tensor(x_train_np, dtype=torch.float32)
    y_train_torch = torch.tensor(y_train_np, dtype=torch.float32)
    x_val_torch = torch.tensor(x_val_np, dtype=torch.float32)
    y_val_torch = torch.tensor(y_val_np, dtype=torch.float32)

    # Convert weights to PyTorch tensors with gradient tracking
    weights_torch = [torch.tensor(W, requires_grad=True) for W in weights]



    for epoch in range(epochs):
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, x_train_torch.shape[0])

            x_batch = x_train_torch[start_idx:end_idx]
            y_batch = y_train_torch[start_idx:end_idx]

            # Forward pass
            y_pred = forward_pass(x_batch, weights_torch, activations)

            # Compute loss
            loss = compute_loss(y_pred, y_batch, loss_func)

            # Backward pass and weight update
            backward_pass(loss, weights_torch, alpha)

        # Compute validation error after each epoch
        if epochs > 0:
            error, _ = compute_validation_error(x_val_torch, y_val_torch, weights_torch, activations)
            errors.append(error)

    # Final output on validation set
    _, y_pred_val_np = compute_validation_error(x_val_torch, y_val_torch, weights_torch, activations)

    # Prepare weight matrices to return
    weight_matrices = [W.detach().numpy() for W in weights_torch]

    if epochs == 0:
        errors = []

    return [weight_matrices, errors, y_pred_val_np]
