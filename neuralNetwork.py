import tensorflow as tf
import numpy as np
import pandas as pd


def next_batch(num, train_data, labels):
    idx = np.arange(0, len(train_data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [train_data[j] for j in idx]
    labels_shuffle = [labels[j] for j in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)


if __name__ == "__main__":
    # TODO: Ensure all steps performed
    np.random.seed(0)

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
    # one_hot = pd.get_dummies(data['class'])

    # Linear Min-Max Scaling
    data_num = data.select_dtypes(include=[np.number])
    minimum = data_num.min()
    data_num = (data_num - minimum)/(data_num.max() - minimum)
    data[data_num.columns] = data_num
    one_hot = pd.get_dummies(data['class'], dtype=np.float32)

    x_values = data_num.values
    y_values = one_hot.values

    total_patterns = y_values.shape[0]

    train_set_size = int(total_patterns * 0.80)
    test_size = int(total_patterns - train_set_size)

    indices_array = np.arange(0, total_patterns)
    np.random.shuffle(indices_array)
    train_indices = indices_array[:train_set_size]
    test_indices = indices_array[train_set_size:]
    x_train = [x_values[index] for index in train_indices]
    y_train = [y_values[index] for index in train_indices]
    x_test = [x_values[index] for index in test_indices]
    y_test = [y_values[index] for index in test_indices]

    # Python optimisation variables
    learning_rate = 0.5
    epochs = 50
    batch_size = 16

    # declare the training data placeholders
    x = tf.placeholder(tf.float32, [None, 11])
    y = tf.placeholder(tf.float32, [None, 3])

    W1 = tf.Variable(tf.random_normal([11, 50], stddev=0.03), name='W1')
    b1 = tf.Variable(tf.random_normal([50]), name='b1')
    W2 = tf.Variable(tf.random_normal([50, 3], stddev=0.03), name='W2')
    b2 = tf.Variable(tf.random_normal([3]), name='b2')

    # calculate the output of the hidden layer
    hidden_out1 = tf.add(tf.matmul(x, W1), b1)
    hidden_out1 = tf.nn.relu(hidden_out1)

    # now calculate the hidden layer output - in this case, let's use a softmax activated
    # output layer
    y_ = tf.nn.softmax(tf.add(tf.matmul(hidden_out1, W2), b2))

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
        total_batch = int(train_set_size / batch_size)
        for epoch in range(epochs):
            avg_cost = 0
            for i in range(total_batch):
                batch_x, batch_y = next_batch(batch_size, x_train, y_train)
                _, c = sess.run([optimiser, cross_entropy], feed_dict={x: batch_x, y: batch_y})
                avg_cost += c / total_batch
            print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(avg_cost))

        print("\nTraining complete!")
        print(sess.run(accuracy, feed_dict={x: np.asarray(x_test), y: np.asarray(y_test)}))
