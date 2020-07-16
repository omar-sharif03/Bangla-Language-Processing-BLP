from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import sys

import pickle
import tensorflow as tf
import numpy as np

import warnings
warnings.filterwarnings("ignore")
from skimage import io, transform
import cv2
from skimage import img_as_uint

import editdistance
import get_feature_vector as gfv;

from sklearn.model_selection import train_test_split

from skimage import morphology, util
from skimage import color
from skimage import filters
from skimage import morphology as mp;
import AdjustImage as ai

LOGDIR = "model_512hidden_3layer_4dropout_72accu/"

#restore the input array and label array form file
myFilei = "train_input.txt";
myFileo = "train_output.txt";
myFiletsti = "test_input.txt";
myFiletsto = "test_output.txt";
with open(myFilei, 'rb') as f:
    train_input = pickle.load(f);
    print("Train input SuccessFully Loaded");
with open(myFileo, 'rb') as f:
    train_output = pickle.load(f);
    print("Train output SuccessFully Loaded");
with open(myFiletsti, 'rb') as f:
    test_input = pickle.load(f);
    print("Test input SuccessFully Loaded");
with open(myFiletsto, 'rb') as f:
    test_output = pickle.load(f);
    print("Test output SuccessFully Loaded");

print(len(train_output));
print(len(test_output));
print(len(test_output)+len(train_output));
#print(train_output);

def toSparse(output_label):
    #put ground truth text into sparse tuple for ctc_loss

    indices = [];
    values = [];
    shape = [len(output_label), 0] #shape[1] must be max(labelList)

    #go over all batches
    for (batchElement, label) in enumerate(output_label):
        #get list of labels
        labels = [c for c in label];

        #sparse tensor must have size of max labels
        if(len(labels) > shape[1]):
            shape[1] = len(labels)

        #put each label on sparse tensor
        for(i, lbl) in enumerate(labels):
            indices.append([batchElement, i]);
            values.append(lbl);

    return (indices, values, shape);

myFile = 'borno.txt';
with open(myFile , 'rb') as f:
    rs_borno = pickle.load(f);

#sort the arrayz
rs_borno.sort();
#print(rs_borno);

def to_string(ar, rs_borno):
    #print(ar);
    ans = "";

    #ans2 = list();
    vowel_dep = ['া', 'ি', 'ী', 'ু', 'ূ', 'ৃ', 'ে', 'ৈ', 'ো', 'ৌ', 'ং', '্'];
    char_set = list();

    indx = 0 ;
    character = "";
    while(indx<len(ar)):
        ch = rs_borno[ar[indx]];

        if(ch != '্'):
            if(len(character) > 0):
                char_set.append(character);
                character = "";

            if(ch not in vowel_dep):
                character += ch;

            else:
                char_set.append(ch);
        else:
            character += ch;
            if(indx+1 < len(ar)):
                indx+=1;
                character += rs_borno[ar[indx]];
        indx+=1;

    if(len(character)>0):
        char_set.append(character);
    #print(char_set);
    ind = 0;
    while(ind<len(char_set)):
        ch = char_set[ind];
        if(ch == 'ে'):
            if(ind+2<len(char_set) and char_set[ind+2] == 'া'):
                ans += char_set[ind+1];
                ans += 'ো';
                ind+=2;
            elif(ind+2<len(char_set) and char_set[ind+2] == 'ী'):
                ans += char_set[ind+1];
                ans += 'ৌ';
                ind+=2;
            else:
                if(ind+1<len(char_set)):
                    ans += char_set[ind+1];
                    ind+=1;
                ans += ch;
        elif(ch == 'ি' or ch =='ৈ'):
            if(ind+1<len(char_set)):
                ans += char_set[ind+1];
                ind+=1;
            ans += ch;
        else:
            ans += ch;
        ind+=1;
    return ans; #, ans2;

#test form http://localhost:8888/notebooks/Normal_practice/Bangla_Processing/Processing%20Bangla%20Word.ipynb
#print(to_string([34, 50, 31, 54, 29, 54, 37, 44, 32, 44, 30, 54, 37, 44, 58], rs_borno));



def lstm_cell(is_training, hidden_size, dropout):
    cell = tf.contrib.rnn.LSTMCell(num_units = hidden_size, forget_bias=1.0, name = 'LSTM');
    if is_training and dropout < 1:
        cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=dropout)
    return cell


class Model(object):
    #constructor
    def __init__(self, is_training, batch_size, num_steps, hidden_size, no_of_output_labels, num_layers, dropout = 0.4, init_scale = 0.05):
        self.is_training = is_training;

        self.batch_size = batch_size;
        self.num_steps = num_steps;
        self.hidden_size = hidden_size;
        self.num_layers = num_layers;
        self.snap_id = 0;

        with tf.name_scope("Input"):
            self.inputs = tf.placeholder(tf.float32, shape=(self.batch_size, self.num_steps, 9));

        with tf.name_scope("Dropout"):
            if(is_training and dropout < 1):
                self.inputs = tf.nn.dropout(self.inputs, dropout);

        #setup Long Short Term Memory Network


        with tf.name_scope("LSTM_cell"):
            #create an LSTM Cell
            cells = [lstm_cell(is_training, hidden_size, dropout) for _ in range(num_layers)] # 2 layers
            # stack basic cells
            stacked = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)

        with tf.name_scope("BLSTM_layer"):
            # bidirectional RNN
            # BxTxF -> BxTx2H
            ((fw, bw), _) = tf.nn.bidirectional_dynamic_rnn(cell_fw=stacked, cell_bw=stacked, inputs=self.inputs, dtype=self.inputs.dtype)

            #concatenate the forward and backward output
            #tf.concat(data, 2) concatenates dimention no. 2
            #tf.expand_dims(data, 2) creates a new dimention at position 2 (done because of CTC ouptup layer)
            #tus BxTxH + BxTxH -> BxTx2H -> BxTx1X2H
            concat_out = tf.expand_dims(tf.concat([fw, bw], 2), 2);



            #reshape the output into BxTx1x2H -> (BxTx1)x2H -> BT1x2H to perform the affine transform
            concat_out = tf.reshape(concat_out, [-1, 2*hidden_size]);


        with tf.name_scope("Affine_Transform"):
            # Perform Affine Transform
            # Truncated normal with mean 0 and std_dev = 0.1   2HxC
            w = tf.Variable(tf.truncated_normal([2*hidden_size, no_of_output_labels+1], stddev = 0.1));
            # Zero bias initialization       C
            b = tf.Variable(tf.constant(0.1, shape = [no_of_output_labels+1]));
            # doing the affine projection BT1x2H * 2HxC + C -> BT1xC
            logits = tf.matmul(concat_out, w) + b;

            tf.summary.histogram('logits', logits);

            # reshaping back to original shape
            # BT1xC -> BxTxC
            logits = tf.reshape(logits, [self.batch_size, self.num_steps, (no_of_output_labels+1)]);



        #no need to use softmax activation because tensorflow CTC loss automatically performs softmax operation
        #setting up CTC
        #create CTC loss and decoder

        #first CTC requires an input of shape[TxBxC] (time major) , so perform a transpose
        # BxTxC -> TxBxC

        with tf.name_scope("Cost_calculation"):
            logits = tf.transpose(logits, [1, 0, 2]);
            self.gtTexts = tf.SparseTensor(
                tf.placeholder(tf.int64, shape = [None, 2]), #indices
                tf.placeholder(tf.int32,shape = [None]),     #Values
                tf.placeholder(tf.int64, shape = [2])        #shape
            )
            #calculate CTC loss for a batch
            self.seqLen = tf.placeholder(tf.int32, [None]) # dimention = batch_size
            loss = tf.nn.ctc_loss(
                labels = self.gtTexts,
                inputs = logits,
                sequence_length = self.seqLen,
                ctc_merge_repeated = True
            )

            #setup cost function for training

            self.cost = tf.reduce_mean(loss);
            tf.summary.scalar('cost', self.cost);

        with tf.name_scope("Decoder"):
            #setup decoder
            self.decoder = tf.nn.ctc_beam_search_decoder(inputs = logits, sequence_length = self.seqLen, beam_width = 80, merge_repeated = False)


        #setup gradient descent optimzer and train operation
        with tf.name_scope("Train"):
            self.batchesTrained = 0
            self.learning_rate = tf.placeholder(tf.float32, shape=[])
            self.train_op = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.cost)


        #setup tensorflow saver and session
        print("Initializing tensorflow");
        print('Python: '+sys.version)
        print('Tensorflow: '+tf.__version__)

        with tf.name_scope("Save_model"):
            self.sess = tf.Session();

            self.saver = tf.train.Saver();
            model_dir = "model/";
            latest_snapshot = tf.train.latest_checkpoint(model_dir);

            self.merged = tf.summary.merge_all();

            if latest_snapshot:
                print("init with stored values from " + latest_snapshot);
                self.saver.restore(self.sess, latest_snapshot);
            else:
                print("init with new values");
                self.sess.run(tf.global_variables_initializer());



    def train_batch(self, batch_input, batch_output):
        #feed batch to nn to train it
        #recay learning rate
        rate = 0.01 if self.batchesTrained < 10 else (0.001 if self.batchesTrained < 10000 else 0.0001)
        (cost, _, summ) = self.sess.run([self.cost, self.train_op, self.merged],feed_dict = {self.inputs : batch_input,
                                                                            self.gtTexts : batch_output,
                                                                            self.seqLen : [self.num_steps] * self.batch_size,
                                                                            self.learning_rate : rate})
        self.batchesTrained += 1;
        return cost, summ;

    def validate_batch(self, batch_input):
        #feed input to rnn and apply decoder
        #initial state

        decoded = self.sess.run([self.decoder], feed_dict = {self.inputs : batch_input,
                                                        self.seqLen : [self.num_steps] * self.batch_size
                                                       });
        return self.to_text(decoded);

    def to_text(self, ctc_output):
        #extracts text output from ctc decoder output

        encodedStr = [[] for _ in range(self.batch_size)];

        decoded = ctc_output[0][0][0] #ctc output returns a tuple(decoded, neg_sum_logits) where decoded is a sparse tensor
        # go over all indices
        #print(decoded.dense_shape);
        for(idx, idx2d) in enumerate(decoded.indices):
            label = decoded.values[idx];

            batch_element = idx2d[0];
            time_step = idx2d[1];

            encodedStr[batch_element].append(label);

        return encodedStr;

    def save(self):
        #save model
        self.snap_id += 1;
        self.saver.save(self.sess, 'model/snapshot', global_step = self.snap_id);

def test_model(batch_size, input_test, output_test):
    m = Model(is_training = False, batch_size = batch_size, num_steps = 64, hidden_size =512, no_of_output_labels = 59, num_layers = 3);
    #evaluate
    train_ind = 0;
    correct_word = 0;
    errors = 0;
    total_char = 0;
    total_word = 0;
    input_word = input_test;
    output_label = output_test;
    while train_ind + batch_size <= len(input_word):
        iw = input_word[train_ind : train_ind + batch_size];
        ol = output_label[train_ind : train_ind + batch_size];

        input_data, output_data = iw, toSparse(ol);

        train_ind += batch_size;

        encodedString = m.validate_batch(input_data);
        #encodedString = m.validate_batch_r(sess, input_data);



        for _ in range(batch_size):
            #calculate error rate
            recog = to_string(encodedString[_], rs_borno);
            orig = to_string(ol[_], rs_borno);
            dist = editdistance.eval(recog, orig);


            print("recognized -> original");
            print(recog," -> ", orig, "    edit distance is : ",dist);

            if(dist==0):correct_word+=1;
            else:
                errors += dist;

            total_char += len(orig);
            total_word += 1;

    error_rate = errors/total_char;
    word_accu = correct_word/total_word;

    print("error rate is ",error_rate," and Word Accuracy is ", word_accu);

test_model(155, test_input, test_output);