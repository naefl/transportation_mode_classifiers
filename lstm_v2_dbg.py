#coding=utf-8
#from tensorflow.python import debug as tf_debug
from itertools import chain
from random import shuffle
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, Imputer
from sklearn import metrics
import pickle
import pandas as pd
import numpy as np
#from numpy import argmax
from scipy import stats
import tensorflow as tf
#from tensorflow.contrib.rnn import GRUCell




#features
features = [
    'speed'
    ,'speedChange'
    ,'bearingsChange'
    ,'accuracy'
    ,'dist1'
    ,'dist2'
    ,'dist3'
    ,'dist4'
    ,'dist5'
    ,'min_time1'
    ,'min_time2'
    ,'min_time3'
    ,'min_time4'
    ,'min_time5'
    ,'accel_max'
    ,'accel_min'
    ,'accel_var'
    ,'peakAverage'
    ,'peakRatio'
    ,'binnedPercentage1'
    ,'binnedPercentage2'
    ,'binnedPercentage3'
    ,'binnedPercentage4'
    ,'binnedPercentage5'
    ,'binnedPercentage6'
    ,'binnedPercentage7'
    ,'binnedPercentage8'
    ,'binnedPercentage9'

    ,'fc2hz'
    ,'fc3hz'
    ,'fc4hz'
    ,'fc5hz'
    ,'fc6hz'
    ,'fft_max_coefficient_freq'
    ,'fft_max_coefficient'
    ,'fc1hzVar'

    ,'ARc2'
    ,'ARc3'
    ,'ARc4'
    ,'ARc5'
    ,'SMA'
]

###############################PARAMS##########################

#config
LOGDIR = '/logs/'

#hyperparams
N_TIME_STEPS = 200
N_HIDDEN_UNITS = 64
N_LAYERS = 1
LEARNING_RATE = 0.02
N_EPOCHS = 70
GRU_INSTEAD_OF_LSTM = True

#params
N_FEATURES = len(features)

step = 1
RANDOM_SEED = 42
N_CLASSES = 6

#size of the internal hidden state
#X_train, X_test, y_train, y_test =  favorite_color = pickle.load( open( "training_test_moving_window_non_nan.p", "rb" ) )

#TODO: replace tf.Variable with tf.get_variable and switch to xavier initialization


#BATCH_SIZE = 1024
BATCH_SIZE = 512


###############################TRANSFORMATION##########################


df = pd.read_csv('combined_trainingsdata.csv').iloc[:,1:]

x= df.iloc[:,:-1].values
imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
imputer = imputer.fit(x)
x = imputer.transform(x)
sc_X = StandardScaler()
df_scaled = sc_X.fit_transform(x)
df_scaled = np.append(df_scaled,df.iloc[:,-1:].values, axis=1)

#shuffling
shuffled_inds = 0
random_sequences_of_min_length = []
while shuffled_inds < len(df_scaled):
    slice = np.random.randint(60,180)
    if not slice + shuffled_inds > len(df_scaled):
        random_sequences_of_min_length.append(df_scaled[shuffled_inds:shuffled_inds+slice])
        shuffled_inds+=slice
    else:
        random_sequences_of_min_length.append(df_scaled[shuffled_inds::])
        shuffled_inds = len(df_scaled)


shuffle(random_sequences_of_min_length)

shuffled_df_with_min_length_seq = list(chain.from_iterable(random_sequences_of_min_length))

#print(len(shuffled_df_with_min_length_seq))








df_scaled = pd.DataFrame(shuffled_df_with_min_length_seq, columns=df.columns, index = df.index)

labels = []
segments = []

 #break down into moving pieces of timesteps
for i in range(0, len(df) - N_TIME_STEPS, step):
  segment = []
  for j in features:
      feature = df[j].values[i: i + N_TIME_STEPS]
      segment.append(feature)
  label = stats.mode(df['label'][i: i + N_TIME_STEPS])[0][0]
  segments.append(segment)
  labels.append(label)



reshaped_segments = np.asarray(segments, dtype= np.float32).reshape( -1, N_TIME_STEPS, N_FEATURES)
labels = np.asarray(pd.get_dummies(labels), dtype = np.float32)





X_train, X_test, y_train, y_test = train_test_split( reshaped_segments, labels, test_size=0.2, shuffle = False, random_state=RANDOM_SEED)

inds = np.where(np.isnan(X_train))
X_train[inds] = np.take(np.nanmean(X_train, axis=0), inds[1])
inds_test = np.where(np.isnan(X_test))
X_test[inds_test] = np.take(np.nanmean(X_test, axis=0), inds_test[1])
pickle.dump([X_train,X_test,y_train,y_test], open("training_test_moving_window_non_nan.p","wb"))
# X_train, X_test, y_train, y_test =   pickle.load( open(         "training_test_moving_window_non_nan.p", "rb" ) )
print(np.where(np.isnan(X_train)))
print(np.where(np.isnan(X_test)))

###############################MODEL##########################

def create_LSTM_model(inputs, name='LSTM'):
    with tf.name_scope(name):
        W = {
            'hidden': tf.Variable(
                tf.random_normal([N_FEATURES, N_HIDDEN_UNITS])
                , name= 'W_hid'),
                'output': tf.Variable(
                    tf.random_normal([N_HIDDEN_UNITS, N_CLASSES], name='W_out')
                )
        }
        biases = {
        'hidden': tf.Variable(
            tf.random_normal([N_HIDDEN_UNITS], mean=1.0), name='B_hid'
        ),
        'output': tf.Variable(
            tf.random_normal([N_CLASSES])
        )
        }
        X = tf.check_numerics(inputs, name="check_input_numerics", message="input was NaN")
        X = tf.transpose(X,[1, 0, 2])
        X = tf.reshape(X, [-1, N_FEATURES])
        hidden = tf.nn.relu(tf.matmul(X, W['hidden']) + biases['hidden'])
        hidden = tf.split(hidden, N_TIME_STEPS, 0)
        # Stack 2 LSTM layers
        if GRU_INSTEAD_OF_LSTM:
            lstm_layer = [tf.contrib.rnn.GRUCell(N_HIDDEN_UNITS) for _ in range(N_LAYERS)]
        else:
            lstm_layer = [tf.contrib.rnn.BasicLSTMCell(N_HIDDEN_UNITS, forget_bias=1.0) for _ in range(N_LAYERS)]
        lstm_layers = tf.contrib.rnn.MultiRNNCell(lstm_layer)
        outputs, _ = tf.contrib.rnn.static_rnn(lstm_layers, hidden, dtype=tf.float32)
        # get output for the last time step
        lstm_last_output = outputs[-1]
        tf.summary.histogram("weights", W['hidden'])
        tf.summary.histogram("weights", W['output'])
        tf.summary.histogram("biases",biases['hidden'])
        tf.summary.histogram("biases",biases['output'])
        ##so we can visualize the distributions of activations coming off this layer
        act = tf.matmul(lstm_last_output, W['output']) + biases['output']
        tf.summary.histogram("activations", act)
        return act

###############################MODEL##########################


###############################DEFINE VARS##########################

tf.reset_default_graph()

X = tf.placeholder(
        tf.float32, [None, N_TIME_STEPS, N_FEATURES], name="input"
    )
Y = tf.placeholder(tf.float32, [None, N_CLASSES])

###############################DEFINE OPS##########################

pred_Y = create_LSTM_model(X, name = 'LSTM')


with tf.name_scope("Prediction"):
    pred_softmax = tf.nn.softmax(pred_Y, name="y_")

L2_LOSS = 0.0015
l2 = L2_LOSS *  sum(
        tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables()
)


with tf.name_scope("loss"):
    loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(
                logits = pred_Y, labels = Y
            ), name="xent_loss"
    ) + l2
    tf.summary.scalar("xent_loss", loss)



with tf.name_scope("train"):
    train_step= tf.train.AdamOptimizer(
            learning_rate=LEARNING_RATE
    ).minimize(loss)


with tf.name_scope("accuracy"):
    correct_pred = tf.equal(tf.argmax(pred_softmax, 1), tf.argmax(Y, 1), name="correct_pred")
    acc= tf.reduce_mean(tf.cast(correct_pred, dtype=tf.float32))
    tf.summary.scalar("accuracy",acc)

with tf.name_scope("debug"):
    correct_pred = tf.Print(correct_pred, [correct_pred], name="correct_pred", message="correct_preds are")
    #argmax_softmax_pred = tf.Print(argmax(pred_softmax,1),[argmax(pred_softmax,1)],name="argmax_softmax_pred", message="argmax_sfotmax_pred is:" )
    #argmax_Y = tf.Print(argmax(Y,1),[argmax(Y,1)], name="argmax_Y", message="argmax_Y")
    input = tf.Print(X,[X], name="input", message="input was")



sess=tf.Session()
summ = tf.summary.merge_all()
writer1 = tf.summary.FileWriter('writer_train_{}'.format(time.time()))
writer2 = tf.summary.FileWriter('writer_test_{}'.format(time.time()))
writer1.add_graph(sess.graph)


saver = tf.train.Saver()
history = dict(train_loss=[], train_acc=[], test_loss=[], test_acc=[])

#sess = tf_debug.LocalCLIDebugWrapperSession(sess)
# sess = tf_debug.TensorBoardDebugWrapperSession(sess, "LAPTOP-RGFADLGU:6064")
sess.run(tf.global_variables_initializer())

###############################TRAINING##############################

train_count = len(X_train)
for i in range(1, N_EPOCHS + 1):
    for j, (start, end ) in enumerate(zip(
            range(0, train_count, BATCH_SIZE),
            range(BATCH_SIZE, train_count + 1,BATCH_SIZE)
            )
        ):

        sess.run(
                [input,
                   # argmax_Y,argmax_softmax_pred,
                    correct_pred,train_step],
                feed_dict={
            X: X_train[start:end],
            Y: y_train[start:end]
            }
        )



    s, _, acc_train, loss_train = sess.run(

        [
            summ,
            pred_softmax,
            acc,
            loss
        ],
        feed_dict={
            X: X_train,
            Y: y_train
        }
    )

    writer1.add_summary(s, i)

    s, _, acc_test, loss_test = sess.run(
            [summ, pred_softmax, acc, loss],
            feed_dict={ X: X_test, Y: y_test}
    )

    writer2.add_summary(s,i)

    tf.summary.scalar("test_acc", acc_test)
    tf.summary.scalar("test_loss", loss_test)

    history['train_loss'].append(loss_train)
    history['train_acc'].append(acc_train)
    history['test_loss'].append(loss_test)
    history['test_acc'].append(acc_test)

    if i != 1 and i % 10 != 0:
        continue

    print(
            'epoch: {} test accuracy: {} loss: {}'.format(
                i,
                acc_test,
                loss_test
            )
    )

###############################TRAINING##############################



###############################FINAL RESULTS##########################

predictions, acc_final, loss_final = sess.run(
        [
            pred_softmax,
            acc,
            loss
        ],
        feed_dict={
            X: X_test,
            Y: y_test
        }
)


print('final results: accuracy: {} loss: {}'.format(acc_final, loss_final))

###############################FINAL RESULTS##########################
