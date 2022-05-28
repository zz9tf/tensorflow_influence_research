import tensorflow as tf

x = tf.Variable([4,2],dtype=tf.float32)

y_ = tf.Variable([11,9],dtype=tf.float32)
with tf.GradientTape() as tape1:
    with tf.GradientTape() as tape2:
        loss = tf.reduce_sum((y_ - x*x)**2/2)
        print(loss)
    grads = tape2.jacobian(loss, x)
    print(grads)
grads2 = tape1.jacobian(grads, x)
print(grads2)