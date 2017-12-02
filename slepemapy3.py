from __future__ import print_function, division
from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn import metrics
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

STUDENTS_COUNT = 39709

def read_dataset(path):
    data = []
    labels = []
    seq_len = []
    max_seq_len = 0
    file = open(path,'r')
    count = 0
    for line in file:
        questions = []
        correct = []
        line_data = line.split(",")
        max_seq_len = max(max_seq_len, len(line_data))
        if (count % 3) == 0:
            count += 1
            continue
        elif (count % 3) == 1:
            data.append(list(map(lambda x:int(x), line_data)))
            seq_len.append(len(line_data))
            count += 1
        elif (count % 3) == 2:
            labels.append(list(map(lambda x:int(x), line_data)))
            count += 1
        if count >= STUDENTS_COUNT*3:
            break
    add_padding(data, max_seq_len)
    add_padding(labels, max_seq_len)

    return data, labels, seq_len, max_seq_len

def add_padding(data, length):
    for entry in data:
        while(len(entry)<length):
            entry.append(int(0))

class SlepeMapyData(object):
    def __init__(self,path):
        self.data, self.labels, self.seqlen, self.max_seq_len = read_dataset(path)
        self.batch_id = 0
    def next(self, batch_size):
        if self.batch_id == len(self.data):
            self.batch_id = 0
        batch_data = (self.data[self.batch_id:min(self.batch_id +
                                                batch_size, len(self.data))])
        batch_labels = (self.labels[self.batch_id:min(self.batch_id +
                                                batch_size, len(self.data))])
        batch_seqlen = (self.seqlen[self.batch_id:min(self.batch_id +
                                                batch_size, len(self.data))])
        self.batch_id = min(self.batch_id + batch_size, len(self.data))
        return batch_data, batch_labels, batch_seqlen

train_path = "./trainDataset.csv"
test_path = "./testDataset.csv"
### DEBUG
# train_path = "/home/dave/projects/recsys/trainDataset.csv"
# test_path = "/home/dave/projects/recsys/testDataset.csv"
train_set = SlepeMapyData(train_path)
test_set = SlepeMapyData(test_path)

num_epochs = 5
total_series_length = train_set.max_seq_len * STUDENTS_COUNT 
num_steps = train_set.max_seq_len 
state_size = 64 # number of hidden neurons
num_classes = 2 # number of classes
echo_step = 0 # we do not need this, how much we should backpropagate
batch_size = 50
num_batches = total_series_length//batch_size//num_steps
learning_rate = 0.2

#
#  MODEL 1
#
x = tf.placeholder(tf.int32, [None, num_steps])
y = tf.placeholder(tf.int32, [None, num_steps])
seqlen = tf.placeholder(tf.int32, [None])

inputs_series = tf.split(x,num_steps, 1)
rnn_inputs = tf.one_hot(x, num_classes)
labels_series = tf.unstack(y, axis=1)

# Forward passes
cell = tf.nn.rnn_cell.BasicLSTMCell(state_size, state_is_tuple=True)
#possible to add initial state initial_state=init_state,
output, current_state = tf.nn.dynamic_rnn(cell=cell, inputs=rnn_inputs,sequence_length=seqlen, dtype=tf.float32)

with tf.variable_scope('softmax'):
    W = tf.get_variable('W', [state_size, num_classes])
    b = tf.get_variable('b', [num_classes], initializer=tf.constant_initializer(0.0))
logits = tf.reshape(tf.matmul(tf.reshape(output, [-1, state_size]), W) + b,
            [-1, num_steps, num_classes])
predictions_series = tf.nn.softmax(logits)

losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
total_loss = tf.reduce_mean(losses)
train_step = tf.train.AdagradOptimizer(learning_rate).minimize(total_loss)

##
##  MODEL 2
##
# x = tf.placeholder(tf.float32, [None, num_steps])
# y = tf.placeholder(tf.int32, [None, num_steps])

# cell_state = tf.placeholder(tf.float32, [batch_size, state_size])
# hidden_state = tf.placeholder(tf.float32, [batch_size, state_size])
# init_state = tf.nn.rnn_cell.LSTMStateTuple(cell_state, hidden_state)

# rnn_inputs = tf.split(x,num_steps, 1)
# labels_series = tf.unstack(y, axis=1)
# # Forward passes
# cell = tf.nn.rnn_cell.BasicLSTMCell(state_size, state_is_tuple=True)
# #possible to add initial state initial_state=init_state,
# output, current_state = tf.nn.static_rnn(cell=cell, inputs=rnn_inputs, dtype=tf.float32)
# W2 = tf.Variable(np.random.rand(state_size, num_classes),dtype=tf.float32)
# b2 = tf.Variable(np.zeros((1,num_classes)), dtype=tf.float32)

# logits_series = [tf.matmul(state, W2) + b2 for state in output] #Broadcasted addition
# predictions_series = [tf.nn.softmax(logits) for logits in logits_series]

# losses = [tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels) for logits, labels in zip(logits_series,labels_series)]
# total_loss = tf.reduce_mean(losses)

# train_step = tf.train.AdagradOptimizer(0.3).minimize(total_loss)

###### END OF MODELS

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    plt.ion()
    plt.figure()
    plt.show()
    loss_list = []
    
    for epoch_idx in range(num_epochs):
        _current_cell_state = np.zeros((batch_size, state_size))
        _current_hidden_state = np.zeros((batch_size, state_size))
        pred_labels = []
        correct_labels = []
        questions = []
        print("New data, epoch", epoch_idx)

        for step in range(num_batches):
            batchX,batchY, batch_seq = train_set.next(batch_size)

            _total_loss, _train_step, _predictions_series = sess.run(
                [total_loss, train_step,  predictions_series],
                feed_dict={
                    x:batchX,
                    y:batchY,
                    seqlen:batch_seq                   
                })

           
            for batch in batchY:
                correct_labels.extend(batch)
            for batch in batchX:
                questions.extend(batch)
            for something in _predictions_series: #list(map(list, zip(*_predictions_series))): # for model 2
                for something2 in something:
                    pred_labels.append(something2[1])
                     
            loss_list.append(_total_loss)

            if step % 100 == 0:
                print("Step",step, "Loss", _total_loss)
                rmse = sqrt(mean_squared_error(pred_labels, correct_labels))
                print(rmse)

            
            # fpr, tpr, thresholds = metrics.roc_curve(batchY[0], pred_labels, pos_label=1)
            # auc = metrics.auc(fpr, tpr)

            # #calculate r^2
            # r2 = r2_score(batchY[0], pred_labels)


        with open('predictions.txt','a') as f:
            for i in range(len(correct_labels)):
                if questions[i] != 0:
                    f.write('question ID is: %d' % questions[i])
                    f.write("pred:%.2f " % pred_labels[i])
                    f.write("correct:%s" % correct_labels[i])
                    f.write("\n")
       
        print("-----------------------")
        losss, predss = sess.run([_total_loss,_predictions_series],
                    feed_dict={
                        x:test_set.data,
                        y:test_set.labels,
                        seqlen:test_set.seqlen
                    })
        pred_labels = []
        correct_labels = []
        for batch in test_set.labels:
                correct_labels.extend(batch)
        for batch in test_set.data:
            questions.extend(batch)
        for something in predss: #list(map(list, zip(*_predictions_series))): # for model 2
            for something2 in something:
                pred_labels.append(something2[1])
        print(losss)
        rmse = sqrt(mean_squared_error(pred_labels, correct_labels))
        print("Printing RMSE FOR TEST LABELS")
        print(rmse)
