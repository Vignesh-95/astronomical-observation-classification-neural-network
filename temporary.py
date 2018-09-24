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


def selective_learning(sess, train_data, labels):
    data_set_length = len(train_data)
    cost = []
    for item in range(data_set_length):
        cost.append([sess.run(cross_entropy, feed_dict={x: np.asarray([train_data[item]]), y: np.asarray([labels[item]])}),
                     train_data[item], labels[item]])
    cost = sorted(cost, key=lambda x: x[0])
    one_third = int(len(cost)/3)
    cost = cost[:one_third] + cost[-one_third:]
    new_batch_x = []
    new_batch_y = []
    for item in range(len(cost)):
        new_batch_x.append(cost[item][1])
        new_batch_y.append(cost[item][2])
    return new_batch_x, new_batch_y


if __name__ == "__main__":
    # TODO: Ensure all steps performed
    np.random.seed(0)
    tf.set_random_seed(0)

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

    # Python optimisation variables
    # learning_rate = np.power(10, math.log(0.1, 10) * np.random.rand(3))
    learning_rate = 0.5
    epochs = 100
    batch_size = 16
    lamb = 0.00001
    total_hidden_neurons_1 = 50
    weight_stdevs = 0.03
    keep_prob = 0.9

    train_test_split_ratio = 0.8
    validation_train_split_size = 0.2
    total_patterns = y_values.shape[0]
    train_set_size = int(total_patterns * train_test_split_ratio)
    test_set_size = int(total_patterns - train_set_size)
    validation_set_size = int(train_set_size * validation_train_split_size)
    train_set_size = train_set_size - validation_set_size
    total_input_dimensions = len(data_num.columns)
    total_output_dimensions = len(one_hot.columns)

    indices_array = np.arange(0, total_patterns)
    np.random.shuffle(indices_array)
    train_indices = indices_array[:train_set_size]
    validation_indices = indices_array[train_set_size:train_set_size + validation_set_size]
    test_indices = indices_array[train_set_size + validation_set_size:]
    x_train = [x_values[index] for index in train_indices]
    y_train = [y_values[index] for index in train_indices]
    x_validate = [x_values[index] for index in validation_indices]
    y_validate = [y_values[index] for index in validation_indices]
    x_test = [x_values[index] for index in test_indices]
    y_test = [y_values[index] for index in test_indices]

    # declare the training data placeholders
    x = tf.placeholder(tf.float32, [None, total_input_dimensions])
    y = tf.placeholder(tf.float32, [None, total_output_dimensions])

    W1 = tf.Variable(tf.random_normal([total_input_dimensions, total_hidden_neurons_1],
                                      stddev=weight_stdevs), name='W1')
    b1 = tf.Variable(tf.random_normal([total_hidden_neurons_1]), name='b1')
    W2 = tf.Variable(tf.random_normal([total_hidden_neurons_1, total_output_dimensions],
                                      stddev=weight_stdevs), name='W2')
    b2 = tf.Variable(tf.random_normal([total_output_dimensions]), name='b2')

    # calculate the output of the hidden layer
    hidden_out1 = tf.add(tf.matmul(x, W1), b1)
    hidden_out1 = tf.nn.relu(hidden_out1)

    # Regularization using dropout
    # Mannual
    # dropout_output = np.random.rand(total_input_dimensions, total_hidden_neurons_1)
    # for i in range(dropout_output.shape[0]):
    #     for j in range(dropout_output.shape[1]):
    #         if dropout_output[i][j] > keep_prob:
    #             dropout_output[i][j] = 0
    # hidden_out1 = tf.multiply(hidden_out1, dropout_output)
    # hidden_out1 /= keep_prob
    # Library Function
    hidden_out1 = tf.nn.dropout(hidden_out1, keep_prob)

    # now calculate the hidden layer output - in this case, let's use a softmax activated
    # output layer
    y_ = tf.nn.softmax(tf.add(tf.matmul(hidden_out1, W2), b2))

    # now let's define the cost function which we are going to train the model on
    y_clipped = tf.clip_by_value(y_, 1e-10, 0.9999999)
    cross_entropy = -tf.reduce_mean(tf.reduce_sum(y * tf.log(y_clipped)
                                                  + (1 - y) * tf.log(1 - y_clipped), axis=1))

    # TODO: Try different regularization schemes
    # Weight Decay Regularization
    # regularization = (lamb/2) * (tf.reduce_sum(tf.square(W1)) + tf.reduce_sum(tf.square(W2)))
    # cross_entropy = cross_entropy + regularization

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
            sx, sy = selective_learning(sess,  x_train, y_train)
            for i in range(total_batch):
                batch_x, batch_y = next_batch(batch_size, sx, sy)
                _, c = sess.run([optimiser, cross_entropy], feed_dict={x: batch_x, y: batch_y})
                avg_cost += c / total_batch
            print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(avg_cost))

        print("\nTraining complete!")
        keep_prob = 1
        print(sess.run(accuracy, feed_dict={x: np.asarray(x_test), y: np.asarray(y_test)}))