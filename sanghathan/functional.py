import numpy as np
def softmax_gradeff(grad_output, output):
    """
    More efficient implementation of softmax gradient without explicit loops.
    Uses the same mathematical foundation but with vectorized operations.
    """
    # Element-wise multiplication of softmax outputs and upstream gradients
    element_mul = output * grad_output
    
    # Sum for each example across features
    sum_term = np.sum(element_mul, axis=1, keepdims=True)
    
    # Final gradient
    return element_mul - output * sum_term

def relu(input):
    return input.clip(min=0.0)

def relu_grad(grad_output , bitmask):
    assert bitmask.dtype == bool 
    return grad_output*bitmask

def linear(input, weight, bias):
    """
    y = x@A^T + b
    """
    return input @ weight.T + bias

def linear_grad(grad_output, input, weight):
    return grad_output @ weight, grad_output.T @ input, grad_output.sum(axis=0)


def softmax(input):
    # logsumexp trick
    input_exp = np.exp(input - np.max(input))
    return input_exp / (input_exp.sum(axis=1, keepdims=True) + 1e-7)


def softmax_grad(grad_output, input):
    # ideally we would cache the output instead of the input during FW,
    # avoiding the recomputation
    output = softmax(input)
    new_grad = output * grad_output
    return new_grad - output * new_grad.sum(axis=-1, keepdims=True)


def mse_loss(input, target, batch_size: int):
    assert input.shape == target.shape
    return ((target - input) ** 2).sum() / batch_size


def mse_loss_grad(input, target, batch_size: int):
    return -2 * (target - input) / batch_size

