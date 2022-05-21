import tensorflow as tf

def hessian_vector_product(xs, function, x0s):
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
      xs: A list of tensors that we should construct the Hessian over.
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

      first_grads = [tf.convert_to_tensor(grad)[0,:] \
                    if tf.convert_to_tensor(grad).shape == 2 \
                    else tf.convert_to_tensor(grad)[0]
                    for grad in first_grads]
      
      elemwise_products = [grad * tf.stop_gradient(x0) for grad, x0 in zip(first_grads, x0s)]
    second_grads = second_backprop.gradient(elemwise_products, xs)

    second_grads = [tf.convert_to_tensor(grad)[0,:] \
                    if tf.convert_to_tensor(grad).shape == 2 \
                    else tf.convert_to_tensor(grad)[0]
                    for grad in second_grads]

    return second_grads
