import sys
sys.path.insert(0, '/Users/raghr010/anaconda/lib/python2.7/site-packages')

import tensorflow as tf
import numpy as np
print np.__file__
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)


batch_size = 100


# a = tf.constant(3.0, dtype=tf.float32)
# b = tf.constant(4.0)
#
# c = tf.Variable(np.random.rand(5,5), dtype=tf.float64)
#
#
#
# print a,b
#
# print c
#
# sess = tf.Session()
#
# print sess.run([a,b])
#
# d = tf.add(a,b)
#
# print sess.run(d)
#
# print d
#
# f = tf.placeholder(tf.int32)
# g = tf.placeholder(tf.int32)
#
# h = f + g
# print h
# print sess.run(h, {f : np.random.randint(5, 784), g : np.random.randint(5, 784)})
# print h
# print sess.run(h, {f : np.ones((5, 1)), g : np.ones((5, 1))})
# print h
#
# i = h * 3
#
# print sess.run(i, {f : np.ones((5, 1)), g : np.ones((5, 1))})
#
#
# W = tf.Variable([0.5], dtype=tf.float32)
# b = tf.Variable([0.5], dtype=tf.float32)
# x = tf.placeholder(tf.float32)
# linear_model = W * x + b
#
# y = tf.placeholder(tf.float32)
# squared_deltas = tf.square(linear_model - y)
# loss = tf.reduce_sum(squared_deltas)
#
# optimizer = tf.train.GradientDescentOptimizer(0.001)
# train = optimizer.minimize(loss)
#
# init = tf.global_variables_initializer()
# sess.run(init)
#
# print 'loss', loss
# print 'train', train
#
# print(sess.run([W, b]))
#
# X_train = np.arange(50,55.0)
# #X_train = X_train / np.max(X_train)
#
# Y_train = np.arange(50.0,55.0)
# #Y_train = Y_train / np.max(Y_train)
#
# print X_train, Y_train

# for i in range(100):
#     sess.run(train, {x : X_train, y : Y_train})
#     print(sess.run([W, b, loss], {x : X_train, y : Y_train}))
#
# print sess.run(loss, {x : X_train, y : Y_train})
#
# fixW = tf.assign(W, [1.])
# fixb = tf.assign(b, [0])
# sess.run([fixW, fixb])
#
# print sess.run(linear_model, {x : X_train})

#print sess.run(loss, {x : range(100), y : range(100)})

# print sess.graph.get_tensor_by_name('add:0')
#
# sum = tf.reduce_sum(linear_model)
#
# a = tf.constant([4,1,2,3])
#
# print  sess.run(tf.argmax(a))
#
# print sess.run(sum, {x : range(100)})


features = [tf.contrib.layers.real_valued_column("x", dimension=1)]
estimator = tf.contrib.learn.LinearRegressor(feature_columns=features)

x_train = np.array([1., 2., 3., 4.])
y_train = np.array([1,2,3,4])
x_eval = np.array([10,11,12])
y_eval = np.array([10,11,12])

input_fn = tf.contrib.learn.io.numpy_input_fn({"x":x_train}, y_train,
                                              batch_size=2,
                                              num_epochs=1000)

eval_input_fn = tf.contrib.learn.io.numpy_input_fn(
    {"x":x_eval}, y_eval, batch_size=4, num_epochs=1000)
estimator.fit(input_fn=input_fn, steps=10000000)

train_loss = estimator.evaluate(input_fn=input_fn)

print train_loss

# Declare list of features, we only have one real-valued feature
def model(features, labels, mode):
  # Build a linear model and predict values
  W = tf.get_variable("W", [1], dtype=tf.float64)
  b = tf.get_variable("b", [1], dtype=tf.float64)
  y = W*features['x'] + b
  # Loss sub-graph
  loss = tf.reduce_sum(tf.square(y - labels))
  # Training sub-graph
  global_step = tf.train.get_global_step()
  optimizer = tf.train.GradientDescentOptimizer(0.01)
  train = tf.group(optimizer.minimize(loss),
                   tf.assign_add(global_step, 1))
  # ModelFnOps connects subgraphs we built to the
  # appropriate functionality.
  return tf.contrib.learn.ModelFnOps(
      mode=mode, predictions=y,
      loss=loss,
      train_op=train)

PIXELS = 28
INPUT_SIZE = PIXELS * PIXELS
NUMBER_OF_CLASSES = 10

import math
def nn_model(images, hidden_unit1, hidden_unit2):

    weights1 = tf.Variable(tf.cast(tf.truncated_normal([INPUT_SIZE, hidden_unit1],
                        stddev=1.0 / math.sqrt(float(INPUT_SIZE))), tf.float64), name='weights1')
    biases1 = tf.Variable(np.zeros((hidden_unit1)), dtype=tf.float64, name='biases1')

    weights2 = tf.Variable(tf.cast(tf.truncated_normal([hidden_unit1, hidden_unit2],
                        stddev=1.0 / math.sqrt(float(hidden_unit1))), dtype=tf.float64), name='weights2')

    biases2 = tf.Variable(np.zeros((hidden_unit2)), dtype=tf.float64, name='biases2')

    weights3 = tf.Variable(tf.cast(tf.truncated_normal([hidden_unit2, NUMBER_OF_CLASSES],
                        stddev=1.0 / math.sqrt(float(hidden_unit2))), dtype=tf.float64), name='weights3')

    biases3 = tf.Variable(np.zeros((NUMBER_OF_CLASSES)), dtype=tf.float64, name='biases3')


    hidden1 = tf.nn.relu(tf.matmul(images, weights1) + biases1)
    hidden2 = tf.nn.relu(tf.matmul(hidden1, weights2) + biases2)

    output = tf.matmul(hidden2, weights3) + biases3

    return output




def train_neural_network():

    x = tf.placeholder(dtype=tf.float64)
    y = tf.placeholder(dtype=tf.float64)

    prediction = nn_model(x, 64, 32)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))

    optimizer = tf.train.AdamOptimizer(1e-4).minimize(cost)

    hm_epochs = 10
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # for epoch in range(hm_epochs):
        #     epoch_loss = 0
        #     for _ in range(int(mnist.train.num_examples/batch_size)):
        #         epoch_x, epoch_y = mnist.train.next_batch(batch_size)
        #         _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
        #         epoch_loss += c

        epoch_loss = 0
        for i in range(20000):
            epoch_x, epoch_y = mnist.train.next_batch(50)
            _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
            epoch_loss += c

        print('Epoch', i, 'completed out of', 1000, 'loss:', epoch_loss)

        correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        print(sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))

        print sess.run(prediction, {x: [mnist.test.images[0]]})


train_neural_network()

def logistic_model(x):
    W = tf.Variable(tf.cast(tf.zeros([784, 10]), tf.float64))
    b = tf.Variable(tf.cast(tf.zeros([10]), tf.float64))

    return tf.nn.softmax(tf.matmul(x, W) + b)

def train_logistic_model():
    x = tf.placeholder(dtype=tf.float64)
    y = tf.placeholder(dtype=tf.float64)

    prediction = logistic_model(x)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))

    optimizer = tf.train.AdamOptimizer().minimize(cost)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        epoch_loss = 0
        for i in range(1000):
            epoch_x, epoch_y = mnist.train.next_batch(batch_size)
            _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
            epoch_loss += c

        print('Epoch', i, 'completed out of',1000,'loss:',epoch_loss)


        correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        print(sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))

        print sess.run(prediction, {x : [mnist.test.images[0]]})

#train_logistic_model()