from __future__ import print_function
import tensorflow as tf
from class_enum import ClassEnum

class_type = ClassEnum.COMPLIANCE_AND_RISK

fg_data_files = ["fg-data-earlest-13Apr17.csv","fg-data-latest-13Apr17.csv"]
logs_path = './tensorflow_logs/fg-' + class_type.name
learning_rate = 0.05
fg_data_size = 600000  # total rows
data_batch_size = 1000
display_step = 10
training_epochs = int(fg_data_size / data_batch_size)

# Network Parameters
n_hidden_1 = 100  # 1st layer number of features
n_hidden_2 = 100  # 2nd layer number of features
n_input = 4  # This must match the number of columns in the features stack (see read_from_csv() ~line 47)
n_classes = class_type.result_classes  # compliant, not_compliant, at_risk, not_at_risk

# tf Graph Input
x = tf.placeholder(tf.float32, [None, n_input], name='InputData')
y = tf.placeholder(tf.float32, [None, n_classes], name='LabelData')

# Length of a file in lines (rows)
def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1


def const_v(val):
    return tf.constant(val, tf.float32)


def const_1():
    return const_v(1)


def const_0():
    return const_v(0)


def both_conditions(true_cond):
    return tf.stack([true_cond, tf.cond(tf.equal(true_cond, const_1()), const_0, const_1)])


def cond_and(cond_1, cond_2, val_1, val_2):
    return tf.logical_and(tf.equal(cond_1, val_1),
                          tf.equal(cond_2, val_2))


def vec4(c1, c2, c3, c4):
    return [const_v(c1), const_v(c2), const_v(c3), const_v(c4)]


# c1 c2 vector
# 1 1 [1, 0 , 0 ,0]
# 1 0 [0, 1, 0, 0]
# 0 1 [0, 0, 1, 0]
# 0 0 [0, 0, 0, 1]

def combined_conditions(cond_1, cond_2):
    return tf.stack(tf.case({cond_and(cond_1, cond_2, const_1(), const_1()): lambda: vec4(1, 0, 0, 0),
                    cond_and(cond_1, cond_2, const_1(), const_0()): lambda: vec4(0, 1, 0, 0),
                    cond_and(cond_1, cond_2, const_0(), const_1()): lambda: vec4(0, 0, 1, 0),
                    cond_and(cond_1, cond_2, const_0(), const_0()): lambda: vec4(0, 0, 0, 1)
                    }, default=lambda: vec4(0, 0, 0, 0), exclusive=True))


def read_from_csv(filename_queue):
    reader = tf.TextLineReader()
    _, csv_row = reader.read(filename_queue)
    # :source, :dob (age), :gender, :duration_minutes, :apnea_hypopnea_index, :apnea_index, :closed_apnea_index, :hypopnea_index,
    # :open_apnea_index, :mask_leak95th_percentile, :respiratory_rate_median, :therapy_mode, :is_compliant, :at_risk
    record_defaults = [[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]]
    col1, col2, col3, col4, col5, col6, col7, col8, col9, col10, col11, col12, is_compliant, at_risk = tf.decode_csv(csv_row,
                                                                        record_defaults=record_defaults)
    # duration, ahi, mask_leak, therapy_mode
    features = tf.stack([col4, col5, col10, col12]) # must match n_input
    if class_type is ClassEnum.RISK:
        print('Using RISK')
        label = both_conditions(at_risk)
    elif class_type is ClassEnum.COMPLIANCE:
        print('Using COMPLIANCE')
        label = both_conditions(is_compliant)
    elif class_type is ClassEnum.COMPLIANCE_AND_RISK:
        print('Using COMPLIANCE_AND_RISK')
        label = combined_conditions(is_compliant, at_risk)
    else:
        raise Exception('Unknown ClassEnum type!!')

    return features, label


def input_pipeline(batch_size, num_epochs=None):
    filename_queue = tf.train.string_input_producer(fg_data_files, num_epochs=num_epochs, shuffle=True)
    example, label = read_from_csv(filename_queue)
    min_after_dequeue = 1000
    capacity = min_after_dequeue + 3 * batch_size
    example_batch, label_batch = tf.train.shuffle_batch(
        [example, label], batch_size=batch_size, capacity=capacity,
        min_after_dequeue=min_after_dequeue)
    return example_batch, label_batch


# Create model
def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['w1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)  # tf.nn.sigmoid(layer_1)
    # Create a summary to visualize the first layer ReLU activation
    tf.summary.histogram("relu1", layer_1)
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['w2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)  # tf.nn.sigmoid(layer_2)
    # Create another summary to visualize the second layer ReLU activation
    tf.summary.histogram("relu2", layer_2)
    # Output layer
    out_layer = tf.add(tf.matmul(layer_2, weights['w3']), biases['b3'])
    return out_layer


# Store layers weight & bias
weights = {
    'w1': tf.Variable(tf.random_normal([n_input, n_hidden_1]), name='W1'),
    'w2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2]), name='W2'),
    'w3': tf.Variable(tf.random_normal([n_hidden_2, n_classes]), name='W3')
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1]), name='b1'),
    'b2': tf.Variable(tf.random_normal([n_hidden_2]), name='b2'),
    'b3': tf.Variable(tf.random_normal([n_classes]), name='b3')
}

# Encapsulating all ops into scopes, making Tensorboard's Graph
# Visualization more convenient
with tf.name_scope('Model'):
    # Build model
    pred = multilayer_perceptron(x, weights, biases)

with tf.name_scope('Loss'):
    # Softmax Cross entropy (cost function)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))

with tf.name_scope('SGD'):
    # Gradient Descent
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    # Op to calculate every variable gradient
    grads = tf.gradients(loss, tf.trainable_variables())
    grads = list(zip(grads, tf.trainable_variables()))
    # Op to update all variables according to their gradient
    apply_grads = optimizer.apply_gradients(grads_and_vars=grads)

with tf.name_scope('Accuracy'):
    # Accuracy
    acc = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    acc = tf.reduce_mean(tf.cast(acc, tf.float32))

# Create a summary to monitor cost tensor
tf.summary.scalar("loss", loss)
# Create a summary to monitor accuracy tensor
tf.summary.scalar("accuracy", acc)
# Create summaries to visualize weights
for var in tf.trainable_variables():
    tf.summary.histogram(var.name, var)
# Summarize all gradients
for grad, var in grads:
    tf.summary.histogram(var.name + '/gradient', grad)
# Merge all summaries into a single op
merged_summary_op = tf.summary.merge_all()

# Initializing the variables
init_op = tf.group(tf.global_variables_initializer(),
                   tf.local_variables_initializer())

# file_length = file_len(fg_data_file)
examples, labels = input_pipeline(data_batch_size)

# scaled_value = (value − min_value) / (max_value − min_value)
with tf.name_scope('Scale'):
    min_example = tf.reduce_min(examples)
    scaled_examples = tf.div(
        tf.subtract(examples, min_example),
        tf.subtract(tf.reduce_max(examples), min_example)
        )

with tf.Session() as sess:
    sess.run(init_op)
    # op to write logs to Tensorboard
    summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

    # Start populating the filename queue.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        batch_xs, batch_ys = sess.run([scaled_examples, labels])
        # Run optimization op (backprop), cost op (to get loss value)
        # and summary nodes
        try:
            _, c, summary = sess.run([apply_grads, loss, merged_summary_op],
                                     feed_dict={x: batch_xs, y: batch_ys})
        except tf.errors.InvalidArgumentError:
            print("Failed on Epoch:", '%04d' % (epoch + 1), "(sample count=", '%d' % ((epoch + 1) * data_batch_size),
                  ")")
            break
        # Write logs at every iteration
        summary_writer.add_summary(summary, epoch * data_batch_size)
        # Compute average loss
        avg_cost += c / data_batch_size
        # Display logs per epoch step
        if (epoch + 1) % display_step == 0:
            print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost), "Accuracy=",
                  acc.eval({x: batch_xs, y: batch_ys}))

    print("Optimization Finished!")
    coord.request_stop()
    coord.join(threads)

    print('Run the command line:\n'
          '--> tensorboard --logdir=./tensorflow_logs '
          '\nThen open http://0.0.0.0:6006/ into your web browser')
