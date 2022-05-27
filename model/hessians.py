from matplotlib.pyplot import axis
import tensorflow as tf

def hessian_vector_product(xs, function, ps, id):
  """
    Multiply the Hessian of `ys` wrt `xs` by `v`.
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
      xs (Tensor, float64): A list of tensors that we construct the Hessian over.
      function (Function): A function contructs ys.
      ps (Tensor, float64): A list of tensors that multiply hessian
      id (Tuple of tensor): An id to get paramters' grads of removed point

  Returns:
      List: A list of tensors (or if the list would be length 1, a single tensor)
      containing the product between the Hessian and `v`.
  """
  with tf.GradientTape() as second_backprop:
    second_backprop.watch(xs)

    with tf.GradientTape() as first_backprop:
      first_backprop.watch(xs)
      ys = function()
    # First backprop
    first_grads = first_backprop.gradient(ys, xs)
      
    elemwise_products = [
      tf.reshape(grad, [-1]) * tf.stop_gradient(x)
      for grad, x in zip(first_grads, ps) 
      if grad is not None
    ]

  # Second backprop
  second_grads = [tf.reshape(grad, [-1]) for grad in second_backprop.gradient(elemwise_products, xs)]

  return second_grads


def get_target_param_grad(grads, ids):
  """This function get the target parameters' gradients guided by provided ids

  Args:
      grads (A list of tensor, float64): this represents all parameters produced by gradient
      ids (A tuple of tensor, int32): this represents the index of the target user and item,
        for example (user_id, item_id)

  Returns:
      tensor, float64: This represents a list of all target gradients.
  """
  item_bias_grad, user_bias_grad, item_embedding_grad, user_embedding_grad, global_bias_grad = grads

  item_bias_grad = tf.gather(tf.convert_to_tensor(item_bias_grad), ids[1])
  user_bias_grad = tf.gather(tf.convert_to_tensor(user_bias_grad), ids[0])
  item_embedding_grad = tf.gather(tf.convert_to_tensor(item_embedding_grad), ids[1], axis=0)
  user_embedding_grad = tf.gather(tf.convert_to_tensor(user_embedding_grad), ids[0], axis=0)
  # global_bias_grad = tf.convert_to_tensor(global_bias_grad)

  grads = [item_bias_grad, user_bias_grad, item_embedding_grad, user_embedding_grad] # , global_bias_grad
  return [tf.reshape(grad, [-1]) for grad in grads] 