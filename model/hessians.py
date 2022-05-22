import tensorflow as tf

def hessian_vector_product(xs, function, p, id):
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
      xs: A list of tensors that we construct the Hessian over.
      function: A function contructs ys.
      p: A list of tensors that multiply hessian
      id: An id to get paramters' grads of removed point
    Returns:
      A list of tensors (or if the list would be length 1, a single tensor)
      containing the product between the Hessian and `v`.
    """
    
    
    with tf.GradientTape() as second_backprop:
      second_backprop.watch(xs)

      with tf.GradientTape() as first_backprop:
        first_backprop.watch(xs)
        ys = function()
      # First backprop
      first_grads = first_backprop.gradient(ys, xs)
      
      elemwise_products = [grad * tf.stop_gradient(x0) for grad, x0 in zip(first_grads, p)]

    second_grads = get_target_param_grad(second_backprop.gradient(elemwise_products, xs), id)

    return second_grads


def get_target_param_grad(grads, id):
  
  item_bias_grad, user_bias_grad, item_embedding_grad, user_embedding_grad, global_bias_grad = grads

  item_bias_grad = tf.reshape(tf.convert_to_tensor(item_bias_grad)[id[1]], [-1])
  user_bias_grad = tf.reshape(tf.convert_to_tensor(user_bias_grad)[id[0]], [-1])
  item_embedding_grad = tf.reshape(tf.convert_to_tensor(item_embedding_grad)[id[1], :], [-1])
  user_embedding_grad = tf.reshape(tf.convert_to_tensor(user_embedding_grad)[id[0], :], [-1])
  global_bias_grad = tf.convert_to_tensor(global_bias_grad)

  return [item_bias_grad, user_bias_grad, item_embedding_grad, user_embedding_grad, global_bias_grad]