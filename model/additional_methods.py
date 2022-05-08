import tensorflow as tf
from tensorflow import gradients
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops

def hessian_vector_product(ys, xs, v):
    """Multiply the Hessian of `ys` wrt `xs` by `v`.
    This is an efficient construction that uses a backprop-like approach
    to compute the product between the Hessian and another vector. The
    Hessian is usually too large to be explicitly computed or even
    represented, but this method allows us to at least multiply by it
    for the same big-O cost as backprop.
    Implicit Hessian-vector products are the main practical, scalable way
    of using second derivatives with neural networks. They allow us to
    do things like construct Krylov subspaces and approximate conjugate
    gradient descent.
    Example: if `y` = 1/2 `x`^T A `x`, then `hessian_vector_product(y,
    x, v)` will return an expression that evaluates to the same values
    as (A + A.T) `v`.
    Args:
      ys: A scalar value, or a tensor or list of tensors to be summed to
          yield a scalar.
      xs: A list of tensors that we should construct the Hessian over.
      v: A list of tensors, with the same shapes as xs, that we want to
         multiply by the Hessian.
    Returns:
      A list of tensors (or if the list would be length 1, a single tensor)
      containing the product between the Hessian and `v`.
    Raises:
      ValueError: `xs` and `v` have different length.
    """

    # Validate the input
    length = len(xs)
    if len(v) != length:
        raise ValueError("xs and v must have the same length.")

    # First backprop
    grads = gradients(ys, xs)

    assert len(grads) == length

    elemwise_products = [
        math_ops.multiply(grad_elem, array_ops.stop_gradient(v_elem))
        for grad_elem, v_elem in zip(grads, v) if grad_elem is not None
    ]

    # Second backprop
    grads_with_none = gradients(elemwise_products, xs)
    return_grads = [
        grad_elem if grad_elem is not None \
            else tf.zeros_like(x) \
        for x, grad_elem in zip(xs, grads_with_none)]

    return return_grads


def variable(name, shape, initializer):
    """
    Helper to create an initialized Variable with weight decay.
    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.
    :param name: name of the variable
    :param shape: list of ints
    :param initializer: An Initializer for variable value
    :return: Variable Tensor of variable
    """
    var = tf.Variable(
        initial_value=initializer(shape=shape, dtype=tf.float64),
        name=name)
    return var

def square_loss(y_true, y_pred):
    """
    This method returns sum((y_true - y_pred)**2)
    :param y_true: A tensor, float 64, represents the real value
    :param y_pred: A tensor, float 64, represents the  predict value
    :return: sum((y_true - y_pred)**2)
    """
    return tf.reduce_sum((y_true - y_pred)**2)