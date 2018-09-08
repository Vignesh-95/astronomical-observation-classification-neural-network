import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

if __name__ == "__main__":
    # TODO: Ensure all steps performed

    # Importing Data
    data = pd.read_csv("/home/vignesh/PycharmProjects/SloanDigitalSkySurvey/"
                       "astronomical-observation-classification-neural-network/"
                       "Skyserver_SQL2_27_2018 6_51_39 PM.csv", skiprows=1)

    # TODO: Data Analysis and Exploration - Statistics and Visual Graphs

    # Take Action on Data - Data Filtering
    data.drop(['objid', 'run', 'rerun', 'camcol', 'field', 'specobjid'], axis=1, inplace=True)

    # TODO: Feature Engineering

    # Try Different Feature Scaling and Normalisation Techniques
    # Standardisation/Normalisation/Z-score
    # data_num = data.select_dtypes(include=[np.number])
    # data_num = (data_num - data_num.mean())/data_num.std()
    # data[data_num.columns] = data_num

    # Linear Min-Max Scaling
    data_num = data.select_dtypes(include=[np.number])
    minimum = data_num.min()
    data_num = (data_num - minimum)/(data_num.max() - minimum)
    data[data_num.columns] = data_num

    x_values = data_num.as_matrix()
    y_values = data['class'].as_matrix()

    train_set_size = int(y_values.shape[0] * 0.80)
    test_size = int(y_values.shape[0] - train_set_size)

    # Python optimisation variables
    learning_rate = 0.5
    epochs = 10
    batch_size = 100

    # declare the training data placeholders
    x = tf.placeholder(tf.float32, [None, 11])
    y = tf.placeholder(tf.float32, [None, 3])

    # now declare the weights connecting the input to the hidden layer
    W1 = tf.Variable(tf.random_normal([11, 300], stddev=0.03), name='W1')
    b1 = tf.Variable(tf.random_normal([300]), name='b1')
    # and the weights connecting the hidden layer to the output layer
    W2 = tf.Variable(tf.random_normal([300, 3], stddev=0.03), name='W2')
    b2 = tf.Variable(tf.random_normal([3]), name='b2')

    # calculate the output of the hidden layer
    hidden_out = tf.add(tf.matmul(x, W1), b1)
    hidden_out = tf.nn.relu(hidden_out)

    # now calculate the hidden layer output - in this case, let's use a softmax activated
    # output layer
    y_ = tf.nn.softmax(tf.add(tf.matmul(hidden_out, W2), b2))

    # now let's define the cost function which we are going to train the model on
    y_clipped = tf.clip_by_value(y_, 1e-10, 0.9999999)
    cross_entropy = -tf.reduce_mean(tf.reduce_sum(y * tf.log(y_clipped)
                                                  + (1 - y) * tf.log(1 - y_clipped), axis=1))

    # add an optimiser
    optimiser = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

    # finally setup the initialisation operator
    init_op = tf.global_variables_initializer()

    # define an accuracy assessment operation
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # start the session
    with tf.Session() as sess:
        # initialise the variables
        sess.run(init_op)
        total_batch = int(len(train_set_size) / batch_size)
        for epoch in range(epochs):
            avg_cost = 0
            for i in range(total_batch):
                batch_x, batch_y = mnist.train.next_batch(batch_size=batch_size)
                _, c = sess.run([optimiser, cross_entropy], feed_dict={x: batch_x, y: batch_y})
                avg_cost += c / total_batch
            print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(avg_cost))
            summary = sess.run(merged, feed_dict={x: mnist.test.images, y: mnist.test.labels})
            writer.add_summary(summary, epoch)

        print("\nTraining complete!")
        writer.add_graph(sess.graph)
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))
    print(data)
