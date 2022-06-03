import tensorflow.compat.v1 as tf
from tensorflow.python.ops import math_ops

def hessian_vector_product(ys, xs, p, get_grads=None):
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
  x, p)` will return an expression that evaluates to the same values
  as A `p`.
  Args:
    ys: A scalar value, or a tensor or list of tensors to be summed to
        yield a scalar.
    xs: A list of tensors that we should construct the Hessian over.
    p: A list of tensors, with the same shapes as xs, that we want to
       multiply by the Hessian.
  Returns:
    A list of tensors (or if the list would be length 1, a single tensor)
    containing the product between the Hessian and `p`.
  Raises:
    ValueError: `xs` and `p` have different length.
  """ 
  # Validate the input
  if len(p) != len(xs):
    raise ValueError("xs and v must have the same length.")

  # First backprop
  grads = tf.gradients(ys, xs)
  if get_grads is not None:
    grads = get_grads(grads)

  assert len(grads) == len(xs)

  elemwise_products = [
      math_ops.multiply(grad, tf.stop_gradient(p_elem)) \
      for grad, p_elem in zip(grads, p) if grad is not None
  ]

  # Second backprop  
  grads_with_none = tf.gradients(elemwise_products, xs)
  return_grads = [
      grad_elem
      if grad_elem is not None else tf.zeros_like(x) \
      for x, grad_elem in zip(xs, grads_with_none)
  ]
  if get_grads is not None:
    return_grads = get_grads(return_grads)
  
  return return_grads
