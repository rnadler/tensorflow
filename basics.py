import tensorflow as tf
import numpy as np

data = np.random.randint(1000, size=50)
print(data)

logs_path = './tensorflow_logs/basic'

x = tf.constant(data, name='x')

y = tf.Variable((5 * x**2) - 3*x + 15, name='y')

with tf.Session() as session:
    merged = tf.summary.merge_all()
    # summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
    model =  tf.global_variables_initializer()
    session.run(model)
    print(session.run(y))
