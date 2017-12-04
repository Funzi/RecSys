from __future__ import print_function, division
from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn import metrics
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

STUDENTS_COUNT = 100

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
            seq_len.append(len(line_data)-1)
            count += 1
        elif (count % 3) == 2:
            labels.append(list(map(lambda x:int(x), line_data)))
            count += 1
            
        if count >= STUDENTS_COUNT*3:
            break
    add_padding(data, max_seq_len)
    add_padding(labels, max_seq_len)

    return data, labels, seq_len, max_seq_len -1

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
        batch_target = batch_data
        batch_input_correct = batch_labels
        for i in range(batch_size):
            temp = batch_data[i].copy()
            temp2 = batch_labels[i].copy()
            batch_data[i] = batch_data[i][:-1]
            batch_target[i] = temp[1:]
            batch_labels[i] = batch_labels[i][1:]
            batch_input_correct[i] = temp2[:-1]
        self.batch_id = min(self.batch_id + batch_size, len(self.data))
        return batch_data, batch_labels, batch_target, batch_input_correct, batch_seqlen

#train_path = "./trainDataset.csv"
#test_path = "./testDataset.csv"
### DEBUG
train_path = "/home/dave/projects/datasets/builder_train.csv"
test_path = "/home/dave/projects/datasets/builder_test.csv"
train_set = SlepeMapyData(train_path)
test_set = SlepeMapyData(test_path)

num_epochs = 5
total_series_length = train_set.max_seq_len * STUDENTS_COUNT 
num_steps = train_set.max_seq_len 
state_size = 1000 # number of hidden neurons
num_classes = 1000# number of classes
echo_step = 0 # we do not need this, how much we should backpropagate
batch_size = 50
num_batches = total_series_length//batch_size//num_steps
learning_rate = 0.1
num_skills = 1000
    
#
#  MODEL 1
#
x = tf.placeholder(tf.int32, [None, num_steps])

y = tf.placeholder(tf.float32, [None, num_steps])
seqlen = tf.placeholder(tf.int32, [None])
target = tf.placeholder(tf.int32, [None, num_steps])
target_label = tf.placeholder(tf.float32, [None, num_steps])

inputs_series = tf.split(x,num_steps, 1)
rnn_inputs = tf.one_hot(x, num_skills)
# TODO append y label to input to pass that information to the enxt output
labels_series = tf.unstack(y, axis=1)
# 
# Forward passes
cell = tf.nn.rnn_cell.BasicLSTMCell(state_size, state_is_tuple=True)
# possible to add initial state initial_state=init_state,
output, current_state = tf.nn.dynamic_rnn(cell=cell, inputs=rnn_inputs,sequence_length=seqlen, dtype=tf.float32)
# TODO embedding one how vector tf.nn.embeddinglookup
with tf.variable_scope('softmax'):
    W = tf.get_variable('W', [state_size, num_classes])
    b = tf.get_variable('b', [num_classes], initializer=tf.constant_initializer(0.0))

logits = tf.reshape(tf.matmul(tf.reshape(output, [-1, state_size]), W) + b,
            [-1, num_steps, num_classes])
predictions_series = tf.sigmoid(logits)
#TODO transfer logits to take only ones according to target id in question
#logits = tf.reshape(logits, [batch_size])
selected_logits = tf.gather(logits, 0, axis=2)

losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=target_label, logits=selected_logits)
total_loss = tf.reduce_mean(losses)
# TODO is it training after every step or after whoel sequence? 
# TODO is it training whole final hidden layer or only selected logits ?
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
    loss_list = []
    
    for epoch_idx in range(num_epochs):
        pred_labels = []
        correct_labels = []
        pred_labels_without0 = []
        correct_labels_without0 = []
        questions = []
        print("New data, epoch", epoch_idx)

        for step in range(num_batches):
            batch_X, batch_Y, batch_target_X, batch_target_Y, batch_seq = train_set.next(batch_size)

            _total_loss, _train_step, _predictions_series = sess.run(
                [total_loss, train_step,  predictions_series],
                feed_dict={
                    x:batch_X,
                    y:batch_Y,
                    target:batch_target_X,
                    target_label:batch_target_Y,
                    seqlen:batch_seq                   
                })

           
            for label in batch_Y:
                correct_labels.extend(label)
            for question in batch_X:
                questions.extend(question)
            i = 0
            for predictions in _predictions_series: #list(map(list, zip(*_predictions_series))): # for model 2
                j = 0
                for prediction in predictions:
                    pred_labels.append(prediction[questions[i*(num_steps-1) + j]]) # second prediction is for probability of correct answer
                    j+=1
                i+=1
            
            for i in range(len(correct_labels)):
                if questions[i] != 0:
                    pred_labels_without0.append(correct_labels[i])
                    correct_labels_without0.append(pred_labels[i])       
            
            loss_list.append(_total_loss)

            if step % 100 == 0:
                print("Step",step, "Loss", _total_loss)
                rmse = sqrt(mean_squared_error(pred_labels_without0, correct_labels_without0))
                print("Epoch train RMSE is: ",rmse)
                with open('results.txt','a') as f:
                    f.write("Step %.2f Loss %.2f \n" % (step, _total_loss))
                    f.write("Epoch train RMSE is: %.2f \n" % rmse)
            # TODO
            # fpr, tpr, thresholds = metrics.roc_curve(batchY[0], pred_labels, pos_label=1)
            # auc = metrics.auc(fpr, tpr)

            # #calculate r^2
            # r2 = r2_score(batchY[0], pred_labels)#


        with open('predictions.txt','a') as f:
            for i in range(len(correct_labels)):
                if questions[i] != 0:
                    f.write('question ID is: %d' % questions[i])
                    f.write("pred:%.2f " % pred_labels[i])
                    f.write("correct:%s" % correct_labels[i])
                    f.write("\n")
       
        print("-----------------------")
        print("Calculating test set predictions")
        # test_loss, test_predictions = sess.run([total_loss, predictions_series],
        #             feed_dict={
        #                 x:test_set.data,
        #                 y:test_set.labels,
        #                 seqlen:test_set.seqlen})
        # pred_labels = []
        # correct_labels = []
        # pred_labels_without0 = []
        # correct_labels_without0 = []
        # for label in test_set.labels:
        #         correct_labels.extend(label)
        # for question in test_set.data:
        #     questions.extend(question)
        # for preditions in test_predictions: #list(map(list, zip(*_predictions_series))): # for model 2
        #     for prediction in preditions:
        #         pred_labels.append(prediction[1])
        # for i in range(len(correct_labels)):
        #         if questions[i] != 0:
        #             pred_labels_without0.append(correct_labels[i])
        #             correct_labels_without0.append(pred_labels[i])
        # print("Loss for test set:%.2f" % test_loss)
        # rmse = sqrt(mean_squared_error(pred_labels_without0, correct_labels_without0))
        # print("RMSE for test set:%.2f" % rmse)
