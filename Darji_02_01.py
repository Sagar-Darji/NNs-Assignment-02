# Darji, Sagar Vishnubhai
# 1002_202_201
# 2024_10_06
# Assignment_02_01

import numpy as np
import torch

def multi_layer_nn_torch(x_train, y_train, layers, activations, alpha=0.01, batch_size=32, epochs=0,
                         loss_func='mse', val_split=(0.8, 1.0), seed=7321):
    # Prepare data
    num_samples = x_train.shape[0]
    input_dim = x_train.shape[1]
    output_dim = y_train.shape[1]

    # Adjusted assertion to handle both cases
    if isinstance(layers[0], int):
        assert output_dim == layers[-1], "Output dimension must match the size of the last layer in 'layers'"
    else:
        assert output_dim == layers[-1].shape[1], "Output dimension must match the number of columns in the last weight matrix"

    # Split data into training and validation sets
    start_idx = int(np.floor(val_split[0] * num_samples))
    end_idx = int(np.floor(val_split[1] * num_samples))

    x_val = x_train[start_idx:end_idx]
    y_val = y_train[start_idx:end_idx]

    # training data: remove validation data
    x_train = np.concatenate((x_train[:start_idx], x_train[end_idx:]), axis=0)
    y_train = np.concatenate((y_train[:start_idx], y_train[end_idx:]), axis=0)

    # Initialize weights
    weights = []
    if isinstance(layers[0], int):
        # Random weight initialization
        prev_dim = input_dim
        for i, layer_size in enumerate(layers):
            # Include bias in weight matrix (bias is in the first row)
            np.random.seed(seed)  # Ensure reproducibility
            W = np.random.randn(prev_dim + 1, layer_size).astype(np.float32)
            weights.append(W)
            prev_dim = layer_size
    else:
        # Use provided weight matrices
        weights = layers

    # Convert weights to torch tensors
    weights_tensors = [torch.tensor(W, dtype=torch.float32, requires_grad=True) for W in weights]

    # Activation functions
    def activation_fn(x, func):
        if func.lower() == 'linear':
            return x
        elif func.lower() == 'sigmoid':
            return torch.sigmoid(x)
        elif func.lower() == 'relu':
            return torch.relu(x)
        else:
            raise ValueError("Invalid activation function. Possible activations are: linear, sigmoid, relu")

    # Loss functions
    def compute_loss(y_pred, y_true):
        if loss_func.lower() == 'mse':
            return torch.mean((y_pred - y_true) ** 2)
        elif loss_func.lower() == 'ce':
            # Cross-entropy loss
            eps = 1e-10
            y_pred = torch.softmax(y_pred, dim=1)
            return -torch.mean(torch.sum(y_true * torch.log(y_pred + eps), dim=1))
        elif loss_func.lower() == 'svm':
            # SVM hinge loss
            y_true_binary = y_true.clone()
            y_true_binary[y_true_binary == 0] = -1  # Convert 0 to -1
            margins = 1 - y_true_binary * y_pred
            loss = torch.mean(torch.clamp(margins, min=0))
            return loss
        else:
            raise ValueError("Invalid loss function. Possible losses are: svm, mse, ce")

    # Forward pass function
    def forward_pass(x, weights):
        a = x
        for idx, W in enumerate(weights):
            # Add bias term to input
            ones = torch.ones(a.shape[0], 1, dtype=torch.float32)
            a = torch.cat([ones, a], dim=1)  # Shape: [batch_size, prev_dim + 1]
            z = torch.matmul(a, W)
            a = activation_fn(z, activations[idx])
        return a

    # Prepare training data as torch tensors
    x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    dataset = torch.utils.data.TensorDataset(x_train_tensor, y_train_tensor)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Errors after each epoch
    errors = []

    # Training loop
    for epoch in range(epochs):
        for x_batch, y_batch in data_loader:
            # Reset gradients
            for W in weights_tensors:
                if W.grad is not None:
                    W.grad.zero_()

            # Forward pass
            output = forward_pass(x_batch, weights_tensors)

            # Compute loss
            loss = compute_loss(output, y_batch)

            # Backward pass
            loss.backward()

            # Update weights
            with torch.no_grad():
                for W in weights_tensors:
                    W -= alpha * W.grad

        # Compute error on validation set
        with torch.no_grad():
            x_val_tensor = torch.tensor(x_val, dtype=torch.float32)
            y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
            val_output = forward_pass(x_val_tensor, weights_tensors)
            error = torch.mean(torch.abs(val_output - y_val_tensor)).item()
            errors.append(float(error))

    # Final predictions on validation set
    with torch.no_grad():
        x_val_tensor = torch.tensor(x_val, dtype=torch.float32)
        val_output = forward_pass(x_val_tensor, weights_tensors)
        val_output_np = val_output.numpy()

    # Convert weights back to numpy arrays
    final_weights = [W.detach().numpy() for W in weights_tensors]

    # If epochs == 0, errors list should be empty
    if epochs == 0:
        errors = []

    return final_weights, errors, val_output_np