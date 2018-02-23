

import numpy as np
import pandas as pd
import tensorflow as tf

#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
#mnist = input_data.read_data_sets("/tmp/data/")


##TODO add more parameters for customized optimization, customized layer types, save accuracy##
class Densetf():
    """Helper for Neural Network Construction"""
    loaded = 0
    #input must be numpy arrays
    def __init__(self,n_inputs,numnodes=[20,30],numout=2,X=0,y=0,model=0,extra=[]):
        
        self.n_inputs = n_inputs
        self.numnodes = numnodes
        self.numout = numout
        self.X = X
        self.y = y
        self.model = 0
        self.extra = extra

    def create_model_and_train(self, n_epochs,batch_size):
        
        n_inputs = self.n_inputs

        numlayers = len(self.numnodes)

        tf.reset_default_graph()
        dic = {}

        with tf.name_scope("dnn"):
            X = tf.placeholder(tf.float32, shape=(None, self.n_inputs), name="X")
            y = tf.placeholder(tf.int64, shape=(None), name="y")
            
            hidden0 = tf.layers.dense(X, self.numnodes[0], name="hidden0", activation=tf.nn.relu,reuse=None)
            dic["hidden0"] = hidden0
            
            for i in xrange(1,numlayers):
                prev = dic["hidden"+str(i-1)]
                dic["hidden"+str(i)] = tf.layers.dense(prev, self.numnodes[i], name=("hidden"+str(i)),activation=tf.nn.relu,reuse=None)
            
            logits = tf.layers.dense(dic["hidden0"], self.numout, name="outputs")
        
        with tf.name_scope("loss"):
            xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
            loss = tf.reduce_mean(xentropy, name="loss")

        learning_rate = 0.01
        
        with tf.name_scope("train"):
            optimizer = tf.train.AdamOptimizer(learning_rate)
            training_op = optimizer.minimize(loss)

        with tf.name_scope("eval"):
            correct = tf.nn.in_top_k(logits, y, 1)
            accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

            
        init = tf.global_variables_initializer()
        self.extra = [dic,X,y,logits]
        saver = tf.train.Saver()
        
        with tf.Session() as sess:
            init.run()
            for epoch in range(n_epochs):
                for iteration in range(self.X.shape[0] // batch_size):
                    newindex = iteration*batch_size
                    X_batch = self.X[newindex:newindex+batch_size,:]
                    y_batch = self.y[newindex:newindex+batch_size]
                    sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
                    acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
                    print(iteration)
                print(epoch, "Train accuracy:", acc_train)
            save_path = saver.save(sess, "models/titanicmodel.ckpt")
    
    
    def load_and_predict(self,ckpt,inp):
        
        if NeuralNetHelper.loaded < 1:

            #re make the model
            extra = self.extra
            X = extra[1]
            y = extra[2]
            layers = extra[0]
            logits = extra[3]

            saver = tf.train.Saver()

            with tf.Session() as sess:
                saver.restore(sess, ckpt)
                X_new_scaled = inp # some new input
                Z = logits.eval(feed_dict={X: X_new_scaled})
                y_pred = np.argmax(Z, axis=1)
                print(y_pred)
                return y_pred
                
        else:
            with tf.Session() as sess:
                X_new_scaled = inp # some new images (scaled from 0 to 1)
                Z = logits.eval(feed_dict={X: X_new_scaled})
                y_pred = np.argmax(Z, axis=1)
                print(y_pred)
                return y_pred
                
        NeuralNetHelper.loaded += 1



#How to use
 # n_inputs = 28*28
 # numnodes = [20,20]
 # n_outputs = 10
 # nn = NeuralNetHelper(n_inputs,2,numnodes,n_outputs,mnist.train.images,mnist.train.labels)\n
 # print nn.n_inputs\n
 # nn.create_model_and_train()