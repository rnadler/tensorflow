import tensorflow as tf

# Point 1
x1 = tf.constant(1, dtype=tf.float32)
y1 = tf.constant(2, dtype=tf.float32)
point1 = tf.stack([x1, y1])

# Point 2
x2 = tf.constant(0, dtype=tf.float32)
y2 = tf.constant(-1, dtype=tf.float32)
point2 = tf.stack([x2, y2])

# Combine points into an array
points = tf.transpose(tf.stack([point1, point2]))

ones = tf.ones((1, 2))

parameters = tf.matmul(ones, tf.matrix_inverse(points))

with tf.Session() as session:
    result = session.run(parameters)
    a = result[0][0]
    b = result[0][1]
    print("Equation: y = {a}x + {b}".format(a=a, b=b))