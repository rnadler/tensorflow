import tensorflow as tf

filename = "olympics2016.csv"

features = tf.placeholder(tf.int32, shape=[3], name='features')
country = tf.placeholder(tf.string, name='country')
total = tf.reduce_sum(features, name='total')

printerop = tf.Print(total, [country, features, total], name='printer')

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    with open(filename) as inf:
        # Skip header
        next(inf)
        for line in inf:
            # Read data, using python, into our features
            country_name, code, gold, silver, bronze, total = line.strip().split(",")
            gold = int(gold)
            silver = int(silver)
            bronze = int(bronze)
            # Run the Print ob
            total = sess.run(printerop, feed_dict={features: [gold, silver, bronze], country:country_name})
            print(country_name, total)