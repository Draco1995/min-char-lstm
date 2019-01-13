# -*- coding: utf-8 -*-
# Minimal character-level language model with a LSTM, in Python/numpy/tensorflow
import tensorflow as tf
import numpy as np

#hyperparameters
hidden_size = 100
seq_size = 50
learning_rate = 1e-3
num_of_stacked_lstm = 1
input_path="./input.txt"
batch_size = 1
restore = False
ckpt_name = "/tmp/model.ckpt"

class input_data:
    def __init__(self,txt_path):
        self.data = open(txt_path,'r',encoding='utf8').read()
        self.data = self.data
        print(self.data)
        self.chars = list(set(self.data))
        self.data_size,self.vocab_size = len(self.data),len(self.chars)
        print('data has {} characters, {} unique.'.format(self.data_size,self.vocab_size))
        self.char_to_ix = {ch:i for i, ch in enumerate(self.chars)}
        self.ix_to_char = {i:ch for i, ch in enumerate(self.chars)}

def one_hot(index,length):
    oh = np.zeros(length)
    oh[index] = 1
    return oh

def main(_):
    data = input_data(input_path)
    tf.reset_default_graph()  

    #Start building the graph
    X = tf.placeholder(tf.float32,[batch_size,None,data.vocab_size]) #[batch_size,Seq_siz,vocab_size]
    Y = tf.placeholder(tf.int32,[batch_size,None]) #[batch_size,Seq_size]
    
    rnn_input = tf.layers.dense(X,units = hidden_size) #[batch_size,seq_size,hidden_size]
    
    cells = [tf.contrib.rnn.LSTMBlockCell(hidden_size,forget_bias=1.0) for _ in range(num_of_stacked_lstm)]
    cell = tf.contrib.rnn.MultiRNNCell(cells)
    with tf.variable_scope('Hidden_state'):
        state_variables = []
        state_variables2 = []
        for state_c, state_h in cell.zero_state(batch_size,tf.float32):
            state_variables.append(tf.nn.rnn_cell.LSTMStateTuple(
                    tf.Variable(state_c,trainable=False),
                    tf.Variable(state_h,trainable=False)))
            state_variables2.append(tf.nn.rnn_cell.LSTMStateTuple(
                    tf.Variable(state_c,trainable=False),
                    tf.Variable(state_h,trainable=False)))
        # Return as a tuple, so that it can be fed to dynamic_rnn as an initial state
        rnn_tuple_state = tuple(state_variables)
        rnn_tuple_state_backup = tuple(state_variables2)
    
    #Bulid the RNN
    with tf.name_scope('LSTM'):
        rnn_output, new_states = tf.nn.dynamic_rnn(cell, rnn_input,
                                                   initial_state = rnn_tuple_state)
    
    transform_outputs = tf.reshape(rnn_output,[-1,hidden_size])
    
    dense_output = tf.layers.dense(transform_outputs,units = data.vocab_size)
    # probs [batch_size,seq_length,data_size]
    probs = tf.reshape(dense_output,[-1,seq_size,data.vocab_size])
    
    # Y [batch_size,seq_length]
    loss_p = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=probs,labels = Y)
    loss = tf.reduce_sum(loss_p,axis = 1)
    samples =  tf.reshape(tf.multinomial(tf.log(tf.nn.softmax(dense_output)),1),[1,1,1])
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
    train_op = optimizer.minimize(loss)
    
    #Define an op to keep the hidden state between batches
    update_ops = []
    for state_variable, new_state in zip(rnn_tuple_state, new_states):
        # Assign the new state to the state variables on this layer
        update_ops.extend([state_variable[0].assign(new_state[0]),
                           state_variable[1].assign(new_state[1])])
    # Return a tuple in order to combine all update_ops into a single operation
    # The tuple's actual value should not be used
    rnn_keep_state_op = tf.tuple(update_ops)


    # Define an op to reset the hidden state to zeros
    update_ops = []
    for state_variable in rnn_tuple_state:
        # Assign the new state to the state variables on this layer
        update_ops.extend([state_variable[0].assign(tf.zeros_like(state_variable[0])),
                           state_variable[1].assign(tf.zeros_like(state_variable[1]))])
    # Return a tuple in order to combine all update_ops into a single operation.
    # The tuple's actual value should not be used.
    rnn_state_zero_op = tf.tuple(update_ops)
    
    # Define an op to copy the hidden state to  backup
    update_ops = []
    for a,b in zip(rnn_tuple_state_backup,rnn_tuple_state):
        # Assign the new state to the state variables on this layer
        update_ops.extend([a[0].assign(b[0]),
                           a[1].assign(b[1])])
    # Return a tuple in order to combine all update_ops into a single operation.
    # The tuple's actual value should not be used.
    rnn_state_backup_op = tf.tuple(update_ops)
    
    # Define an op to restore the hidden state from backup
    update_ops = []
    for a,b in zip(rnn_tuple_state,rnn_tuple_state_backup):
        # Assign the new state to the state variables on this layer
        update_ops.extend([a[0].assign(b[0]),
                           a[1].assign(b[1])])
    # Return a tuple in order to combine all update_ops into a single operation.
    # The tuple's actual value should not be used.
    rnn_state_restore_op = tf.tuple(update_ops)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        n,p = 0,0
        smooth_loss = -np.log(1.0/data.vocab_size)*seq_size
        if restore == True:
            saver.restore(sess, ckpt_name)
            n,p,smooth_loss = np.load(ckpt_name+".npy")
            n = int(n)
            p = int(p)
            print("Model restored, starting from n={},p={},smooth_loss={}".format(n,p,smooth_loss))
        
        while n<300000:
            if p+seq_size+1 >= data.data_size or n==0:
                sess.run(rnn_state_zero_op)
                p = 0
                
            inputs = [[one_hot(data.char_to_ix[ch],data.vocab_size) for ch in data.data[p:p+seq_size]]]
            targets = [[data.char_to_ix[ch] for ch in data.data[p+1:p+seq_size+1]]]

            if n%99 == 0:
                
                sess.run(rnn_state_backup_op)
                inputs2 = [[one_hot(data.char_to_ix[ch],data.vocab_size) for ch in data.data[p:p+1]]]

                chars = []
                for i in range(seq_size*4):
                    sam,_ = sess.run([samples,rnn_keep_state_op],feed_dict={X:inputs2})
                    sam = sam[0]
                    chars.append(sam[0])
                    inputs2[0][0] = one_hot(sam[0][0],data.vocab_size)
                    
                txt = ''.join(data.ix_to_char[ix[0]] for ix in chars)
                sess.run(rnn_state_restore_op)
                saver.save(sess,ckpt_name)
                array = []
                array.append(n)
                array.append(p)
                array.append(smooth_loss)
                np.save(ckpt_name+".npy",array)
                print("----\n {} \n----".format(txt))
                
            loss_,_,_ = sess.run([loss,train_op,rnn_keep_state_op],feed_dict={X:inputs,Y:targets})
            smooth_loss = smooth_loss * 0.999 + loss_ * 0.001
            if n % 99 == 0: 
                print ('iter %d, p %d, loss: %f' % (n,p, smooth_loss)) # print progress
            p += seq_size
            n += 1

if __name__ == "__main__":
    tf.app.run()