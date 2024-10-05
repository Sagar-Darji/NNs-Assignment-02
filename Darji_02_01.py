# Darji, Sagar Vishnubhai
# 1002_202_201
# 2024_10_06
# Assignment_02_01

import numpy as np
import torch
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader, TensorDataset

def multi_layer_nn_torch(x_train, y_train, layers, activations, alpha=0.01, batch_size=32, epochs=0,
                         loss_func='mse', val_split=(0.8, 1.0), seed=7321):
    import torch

    # Initialize weights
    if isinstance(layers[0], int):
        np.random.seed(seed)
        weights = []
        for i in range(len(layers)):
            if i == 0:
                input_dim = x_train.shape[1]
            else:
                input_dim = layers[i - 1]
            output_dim = layers[i]
            W = np.random.randn(input_dim + 1, output_dim).astype(np.float32)
            weights.append(W)
    else:
        # Use provided weight matrices
        weights = layers

    # Convert weights to PyTorch tensors with gradient tracking
    weights_torch = [torch.tensor(W, requires_grad=True) for W in weights]

    # Prepare data
    x_train = torch.tensor(x_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)

    total_samples = x_train.shape[0]
    val_start = int(np.floor(val_split[0] * total_samples))
    val_end = int(np.floor(val_split[1] * total_samples))

    x_val = x_train[val_start:val_end]
    y_val = y_train[val_start:val_end]

    x_train = x_train[:val_start]
    y_train = y_train[:val_start]

    num_batches = int(np.ceil(x_train.shape[0] / batch_size))
    errors = []

    for epoch in range(epochs):
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, x_train.shape[0])

            x_batch = x_train[start_idx:end_idx]
            y_batch = y_train[start_idx:end_idx]

            # Forward pass
            activations_list = []
            input = x_batch
            for idx, (W, activation) in enumerate(zip(weights_torch, activations)):
                ones = torch.ones(input.shape[0], 1)
                input_with_bias = torch.cat((ones, input), dim=1)
                z = torch.mm(input_with_bias, W)

                if activation.lower() == 'linear':
                    a = z
                elif activation.lower() == 'sigmoid':
                    a = 1 / (1 + torch.exp(-z))
                elif activation.lower() == 'relu':
                    a = torch.clamp(z, min=0)
                else:
                    raise ValueError(f"Unknown activation function: {activation}")
                activations_list.append(a)
                input = a

            y_pred = activations_list[-1]

            # Compute loss
            if loss_func.lower() == 'mse':
                loss = torch.mean((y_pred - y_batch) ** 2)
            elif loss_func.lower() == 'ce':
                exp_scores = torch.exp(y_pred)
                probs = exp_scores / torch.sum(exp_scores, dim=1, keepdim=True)
                correct_logprobs = -torch.log(probs[torch.arange(y_batch.shape[0]), torch.argmax(y_batch, dim=1)] + 1e-15)
                loss = torch.sum(correct_logprobs) / y_batch.shape[0]
            elif loss_func.lower() == 'svm':
                y_true = torch.argmax(y_batch, dim=1)
                batch_size = y_pred.shape[0]
                correct_class_scores = y_pred[torch.arange(batch_size), y_true].view(-1, 1)
                margins = y_pred - correct_class_scores + 1.0
                margins[torch.arange(batch_size), y_true] = 0
                margins = torch.clamp(margins, min=0)
                loss = torch.sum(margins) / batch_size
            else:
                raise ValueError(f"Unknown loss function: {loss_func}")

            # Backward pass
            loss.backward()

            # Update weights
            with torch.no_grad():
                for W in weights_torch:
                    W -= alpha * W.grad
                    W.grad.zero_()

        # Compute validation error
        if epochs > 0:
            input = x_val
            for idx, (W, activation) in enumerate(zip(weights_torch, activations)):
                ones = torch.ones(input.shape[0], 1)
                input_with_bias = torch.cat((ones, input), dim=1)
                z = torch.mm(input_with_bias, W)

                if activation.lower() == 'linear':
                    a = z
                elif activation.lower() == 'sigmoid':
                    a = 1 / (1 + torch.exp(-z))
                elif activation.lower() == 'relu':
                    a = torch.clamp(z, min=0)
                else:
                    raise ValueError(f"Unknown activation function: {activation}")
                input = a

            y_pred_val = input
            error = torch.mean(torch.abs(y_pred_val - y_val)).item()
            errors.append(error)

    # Final output on validation set
    input = x_val
    for idx, (W, activation) in enumerate(zip(weights_torch, activations)):
        ones = torch.ones(input.shape[0], 1)
        input_with_bias = torch.cat((ones, input), dim=1)
        z = torch.mm(input_with_bias, W)

        if activation.lower() == 'linear':
            a = z
        elif activation.lower() == 'sigmoid':
            a = 1 / (1 + torch.exp(-z))
        elif activation.lower() == 'relu':
            a = torch.clamp(z, min=0)
        else:
            raise ValueError(f"Unknown activation function: {activation}")
        input = a

    y_pred_val = input.detach().numpy()

    # Prepare weight matrices to return
    weight_matrices = [W.detach().numpy() for W in weights_torch]

    if epochs == 0:
        errors = []

    return [weight_matrices, errors, y_pred_val]
