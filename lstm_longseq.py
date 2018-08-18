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
from sklearn.externals import joblib
#from tensorflow.contrib.rnn import GRUCell

###############################PARAMS##########################

#config
LOGDIR = '/logs/'

#hyperparams

BATCH_SIZE = 1024
N_TIME_STEPS = 100
N_HIDDEN_UNITS = 64
LEN_TRAIN_SET_SHUFFLE = 600
N_LAYERS = 4
LEARNING_RATE = 0.0005
N_EPOCHS = 70
FRAC_TRAIN = 1
step = 10
GRU_INSTEAD_OF_LSTM = False
ACCEL = True
GYRO = True
GPS = True
SOUND = True
COMPASS = True
XAVIER_INIT = True
TRUNCATED_NORMAL_INIT = False 

print(XAVIER_INIT)
print('XAVIER_INIT')
print(N_TIME_STEPS )
print('N_TIME_STEPS ')
print(N_HIDDEN_UNITS )
print('N_HIDDEN_UNITS ')
print(LEN_TRAIN_SET_SHUFFLE )
print('LEN_TRAIN_SET_SHUFFLE ')
print(TRUNCATED_NORMAL_INIT)
print('TTRUNCATED_NORMAL_INITR')
print(N_LAYERS )
print('N_LAYERS ')
print(LEARNING_RATE )
print('LEARNING_RATE ')
print(N_EPOCHS )
print('N_EPOCHS ')
print(GRU_INSTEAD_OF_LSTM )
print('GRU_INSTEAD_OF_LSTM ')
print(FRAC_TRAIN)
print('FRAC_TRAIN')
print(step)
print('step')
print(N_TIME_STEPS )
print('N_TIME_STEPS ')
print(N_HIDDEN_UNITS )
print('N_HIDDEN_UNITS ')
print(LEN_TRAIN_SET_SHUFFLE )
print('LEN_TRAIN_SET_SHUFFLE ')
print(N_LAYERS )
print('N_LAYERS ')
print(LEARNING_RATE )
print('LEARNING_RATE ')
print(N_EPOCHS )
print('N_EPOCHS ')
print(GRU_INSTEAD_OF_LSTM )
print('GRU_INSTEAD_OF_LSTM ')
print(step)
print('step')
print(BATCH_SIZE)
print('BBATCH_SIZEA')

columns = ['accel_x','accel_y','accel_z','gyro_x','gyro_y','gyro_z','db','azimuth','accuracy','speed','satellites in view','satellites in use','label']
features = columns[:-1]

gps =  [features.index(item) for item in ['accuracy','speed','satellites in view','satellites in use']]
accel = [features.index(item) for item in ['accel_x','accel_y','accel_z']]
gyro = [features.index(item) for item in ['gyro_x','gyro_y','gyro_z']]
sound = [features.index(item) for item in ['db']]
compass = [features.index(item) for item in ['azimuth']]

#params
N_FEATURES = len(features)

RANDOM_SEED = 42
N_CLASSES = 6

#size of the internal hidden state

#TODO: replace tf.Variable with tf.get_variable and switch to xavier initialization




# ###############################TRANSFORMATION##########################


df = pd.read_csv('comb_dfs_accel_gyro_sound_compass_gps.csv')
print('df.head after loading')
print(df.head(1))
print('df shape after loading')
print(df.shape)
x = df.iloc[:,:-1].values
y = df.iloc[:,-1].values

print("X Shape,describe  after loading")
print(x.shape)
print(y.shape)
print("Labels shape after loading")
print(pd.Series(y).describe())
imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
imputer = imputer.fit(x)

x = imputer.transform(x)
sc_X = StandardScaler()
x_scaled = sc_X.fit_transform(x)
print("X Shape,describe  after X = data.iloc[:,-1]")
print(x_scaled.shape)
print(pd.DataFrame(x_scaled).describe())

df_scaled = np.append(x_scaled,df.iloc[:,-1:].values, axis=1)
print("df_scaled Shape, describe  after X = data.iloc[:,-1]")
print(df_scaled.shape)
print(pd.DataFrame(df_scaled).describe())

#shuffling to get all modes of transport in test set
shuffled_inds = 0
random_sequences_of_min_length = []
while shuffled_inds < len(df_scaled):
    slice = np.random.randint(600,1800)
    if not slice + shuffled_inds > len(df_scaled):
        random_sequences_of_min_length.append(df_scaled[shuffled_inds:shuffled_inds+slice])
        shuffled_inds += slice
    else:
        random_sequences_of_min_length.append(df_scaled[shuffled_inds::])
        shuffled_inds = len(df_scaled)


shuffle(random_sequences_of_min_length)

shuffled_df_with_min_length_seq = np.array(list(chain.from_iterable(random_sequences_of_min_length)))
print("shuffled_df_with_min_length_seq shape")
print(shuffled_df_with_min_length_seq.shape)


df_scaled_train = pd.DataFrame(shuffled_df_with_min_length_seq[:int(np.floor(0.8*len(df_scaled)))], columns=df.columns)
df_scaled_test = pd.DataFrame(shuffled_df_with_min_length_seq[int(np.floor(0.8*len(df_scaled))+1):], columns=df.columns)
df_scaled_train.to_csv('all_sensors_raw_scaled_shuffled_train.csv',index=False)
df_scaled_test.to_csv('all_sensors_raw_scaled_shuffled_test.csv',index=False)
print('done with df')
df_scaled_test = pd.read_csv('all_sensors_raw_scaled_shuffled_test.csv').values
df_scaled_train = pd.read_csv('all_sensors_raw_scaled_shuffled_train.csv').values
print('df_scaled_train shape')
print(df_scaled_train.shape)
print('df_scaled_test shape')
print(df_scaled_test.shape)

#train set (shuffled)

#shuffling
shuffled_inds = 0
random_sequences_of_min_length_train = []
while shuffled_inds < len(df_scaled_train):
    slice = np.random.randint(int(LEN_TRAIN_SET_SHUFFLE/3),LEN_TRAIN_SET_SHUFFLE)
    if not slice + shuffled_inds > len(df_scaled_train):
        random_sequences_of_min_length_train.append(df_scaled_train[shuffled_inds:shuffled_inds+slice])
        shuffled_inds+=slice
    else:
        random_sequences_of_min_length_train.append(df_scaled_train[shuffled_inds:])
        shuffled_inds = len(df_scaled_train)


shuffle(random_sequences_of_min_length_train)
shuffled_df_with_min_length_seq_train = list(chain.from_iterable(random_sequences_of_min_length_train))
shuffled_df_with_min_length_seq_train =   shuffled_df_with_min_length_seq_train[:int(len(df_scaled_train)*FRAC_TRAIN)]

df_scaled_train = pd.DataFrame(shuffled_df_with_min_length_seq_train, columns=features+['label'])
df_scaled_test = pd.DataFrame(df_scaled_test, columns=features+['label'])
print('df_scaled_train shape after shuffle')
print(df_scaled_train.shape)
print('Labels train after train shuffling')
print(df_scaled_train['label'].describe())

labels_train = []
segments_train = []

fails = []

#TRAIN DATASET TO SEQ_LEN

for i in range(0, len(df_scaled_train) - N_TIME_STEPS, step):
    segment_train = []
    for j in features:
        feature = df_scaled_train[j].values[i: i + N_TIME_STEPS]
        segment_train.append(feature)
    label_train = df_scaled_train['label'][i + N_TIME_STEPS]
    segments_train.append(segment_train)
    labels_train.append(label_train)




reshaped_segments_train = np.asarray(segments_train, dtype=np.float32).reshape(-1, N_TIME_STEPS, N_FEATURES)
print('reshaped_segments_train/ x_train shape')
print(reshaped_segments_train.shape)
print('labels train after breakdown into seq_len')
print('{},{},{}'.format(len(labels_train),len(labels_train[0]),len(labels_train[0][0])))
print(set(labels_train))
print(pd.Series(labels_train).describe())
labels_train_dummy = np.asarray(pd.get_dummies(labels_train), dtype = np.float32)
print('labels_train_dummy/ y_train shape')
print(labels_train_dummy.shape)


#test set (unshuffled)


labels_test= []
segments_test= []

 #break down into moving pieces of timesteps

for i in range(0, len(df_scaled_test) - N_TIME_STEPS, step):
  segment_test= []
  for j in features:
      feature = df_scaled_test[j].values[i: i + N_TIME_STEPS]
      segment_test.append(feature)
  label_test= df_scaled_test['label'][i + N_TIME_STEPS]
  segments_test.append(segment_test)
  labels_test.append(label_test)



reshaped_segments_test= np.asarray(segments_test, dtype= np.float32).reshape( -1, N_TIME_STEPS, N_FEATURES)
print('reshaped_segments_test/x_test shape')
print(reshaped_segments_test.shape)
labels_test_dummy= np.asarray(pd.get_dummies(labels_test), dtype = np.float32)
print('labels_test_dummy/y_test shape')
print(labels_test_dummy.shape)

X_train = reshaped_segments_train
X_test = reshaped_segments_test
y_train = labels_train_dummy
y_test = labels_test_dummy

inds = np.where(np.isnan(X_train))
X_train[inds] = np.take(np.nanmean(X_train, axis=0), inds[1])
inds_test = np.where(np.isnan(X_test))
X_test[inds_test] = np.take(np.nanmean(X_test, axis=0), inds_test[1])


inds = np.where(np.isnan(X_train))
X_train[inds] = np.take(np.nanmean(X_train, axis=0), inds[1])
inds_test = np.where(np.isnan(X_test))
X_test[inds_test] = np.take(np.nanmean(X_test, axis=0), inds_test[1])
print('X_train shape after nan checking')
print(X_train.shape)
print('X_test shape after nan checking')
print(X_test.shape)
pickle.dump([X_train,X_test,y_train,y_test], open("training_test_raw_{}_all_sensors.p".format(N_TIME_STEPS),"wb"))
X_train_temp, X_test_temp, y_train_temp, y_test_temp = pickle.load(open("training_test_raw_{}_all_sensors.p".format(N_TIME_STEPS), "rb"))
#X_train_temp = X_train_temp[:int(len(X_train_temp)*FRAC_TRAIN),:]
#y_train_temp = y_train_temp[:int(len(y_train_temp)*FRAC_TRAIN),:]
X_train = X_train_temp[:,:,accel]
X_test = X_test_temp[:,:,accel]
if GYRO:
    X_train, X_test = np.append(X_train, X_train_temp[:,:,gyro], axis=2),np.append(X_test, X_test_temp[:,:,gyro], axis=2)
if COMPASS:
    X_train, X_test = np.append(X_train, X_train_temp[:,:,compass], axis=2),np.append(X_test, X_test_temp[:,:,compass], axis=2)
if SOUND:
    X_train, X_test = np.append(X_train, X_train_temp[:,:,sound], axis=2),np.append(X_test, X_test_temp[:,:,sound], axis=2)
if GPS:
    X_train, X_test = np.append(X_train, X_train_temp[:,:,gps], axis=2),np.append(X_test, X_test_temp[:,:,gps], axis=2)
# # print(np.where(np.isnan(X_train)))
# # print(np.where(np.isnan(X_test)))

###############################MODEL##########################

def create_LSTM_model(inputs, name='LSTM'):
    with tf.name_scope(name):
        if XAVIER_INIT:
            W = {
                'hidden': tf.get_variable(
                    shape=[N_FEATURES, N_HIDDEN_UNITS],
                    name= 'W_hid',
                    initializer=tf.contrib.layers.xavier_initializer()
                ),
                'output': tf.get_variable(
                    shape=[N_HIDDEN_UNITS, N_CLASSES],
                    name='W_out',
                    initializer=tf.contrib.layers.xavier_initializer()
                )
            }
            biases = {
                'hidden': tf.get_variable(
                    shape=[N_HIDDEN_UNITS],
                    name='B_hid',
                    initializer=tf.contrib.layers.xavier_initializer()
                ),
                'output': tf.get_variable(
                    shape=[N_CLASSES],
                    initializer=tf.contrib.layers.xavier_initializer(),
                    name='B_out'
                )
            }
        if TRUNCATED_NORMAL_INIT:
            W = {
                'hidden': tf.Variable(
                    tf.truncated_normal([N_FEATURES, N_HIDDEN_UNITS], stddev=0.8)
                    , name= 'W_hid'),
                'output': tf.Variable(
                    tf.truncated_normal([N_HIDDEN_UNITS, N_CLASSES],stddev=0.8),
                    name='W_out'
                )
            }
            biases = {
                'hidden': tf.Variable(
                    tf.truncated_normal([N_HIDDEN_UNITS], mean=1.0,stddev=0.8), name='B_hid'
                ),
                'output': tf.Variable(
                    tf.truncated_normal([N_CLASSES], stddev=0.8)
                )
            }
        else:
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
        X = tf.transpose(inputs,[1, 0, 2])
        X = tf.reshape(X, [-1, N_FEATURES])
        hidden = tf.nn.relu(tf.matmul(X, W['hidden']) + biases['hidden'])
        hidden = tf.split(hidden, N_TIME_STEPS, 0)
        if GRU_INSTEAD_OF_LSTM:
            lstm_layer = [tf.contrib.rnn.GRUCell(N_HIDDEN_UNITS) for _ in range(N_LAYERS)]
        else:
            lstm_layer = [tf.contrib.rnn.BasicLSTMCell(N_HIDDEN_UNITS, forget_bias=1.0) for _ in range(N_LAYERS)]
        lstm_layers = tf.contrib.rnn.MultiRNNCell(lstm_layer)
        outputs, _ = tf.contrib.rnn.static_rnn(lstm_layers, hidden, dtype=tf.float32)
        lstm_last_output = outputs[-1]
        tf.summary.histogram("weights", W['hidden'])
        tf.summary.histogram("weights", W['output'])
        tf.summary.histogram("biases",biases['hidden'])
        tf.summary.histogram("biases",biases['output'])
        act = tf.matmul(lstm_last_output, W['output']) + biases['output']
        tf.summary.histogram("activations", act)
        return act



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
        tf.nn.softmax_cross_entropy_with_logits(
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

# with tf.name_scope("debug"):
# correct_pred = tf.Print(correct_pred, [correct_pred], name="correct_pred", message="correct_preds are")
#argmax_softmax_pred = tf.Print(argmax(pred_softmax,1),[argmax(pred_softmax,1)],name="argmax_softmax_pred", message="argmax_sfotmax_pred is:" )
#argmax_Y = tf.Print(argmax(Y,1),[argmax(Y,1)], name="argmax_Y", message="argmax_Y")
# input = tf.Print(X,[X], name="input", message="input was")



sess=tf.Session()
summ = tf.summary.merge_all()
writer1 = tf.summary.FileWriter('writer_train_{}_{}_{}_{}_{}'.format(time.time(),N_TIME_STEPS,LEN_TRAIN_SET_SHUFFLE, N_LAYERS, N_EPOCHS))
writer2 = tf.summary.FileWriter('writer_test_{}_{}_{}_{}_{}'.format(time.time(),N_TIME_STEPS,LEN_TRAIN_SET_SHUFFLE, N_LAYERS, N_EPOCHS))
writer1.add_graph(sess.graph)


saver = tf.train.Saver()
history = dict(train_loss=[], train_acc=[], test_loss=[], test_acc=[])

#sess = tf_debug.LocalCLIDebugWrapperSession(sess)
# sess = tf_debug.TensorBoardDebugWrapperSession(sess, "LAPTOP-RGFADLGU:6064")
sess.run(tf.global_variables_initializer())

###############################TRAINING##############################

train_count = len(X_train)
for i in range(1, N_EPOCHS + 1):
    for j, (start, end) in enumerate(zip(
            range(0, train_count, BATCH_SIZE),
            range(BATCH_SIZE, train_count + 1,BATCH_SIZE)
    )
    ):

        sess.run(
            [
                # argmax_Y,argmax_softmax_pred,
                train_step],
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

    # if i != 1 and i % 10 != 0:
    #     continue

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
