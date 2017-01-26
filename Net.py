import tensorflow as tf
import numpy as np

SAMPLESIZE = 6
numFilters = 2

class Net():
    def __init__(self, sampleSize):
        #height is added to accomodate tensorflows lack of support for 1d atrous convolutions 
        #height = 1
        numChannels = 1 
        self.sampleSize = sampleSize
        #None allows us to enter a batch. sampleSize can be an entire song if training or just 1 if generating samples.
        self.input = tf.placeholder(tf.float32, shape = [None,1, self.sampleSize, numChannels])


    def _cust_atrous_conv2d(self, input, dialation, name):
        with tf.variable_scope("atrous_"+str(name)) as scope:
            const = tf.constant([0.0 for i in range(dialation)])
            reshapedInput = tf.reshape(input,(self.sampleSize,))
            inputShape = input.get_shape()[0]
            concated = tf.slice(tf.concat(0,[const, reshapedInput]), [0],[self.sampleSize])
            stacked = tf.concat(3,[tf.reshape(concated,shape=(-1,1,self.sampleSize, 1)), input])
            stacked = tf.concat(3, [stacked for i in range(10)])
            _,_,_,prevChannels = stacked.get_shape()
            print(prevChannels)
            output = tf.nn.conv2d(stacked, self._weight_variable([1,1,int(prevChannels),numFilters]), strides=[1,1,1,1], padding = "SAME")
            return stacked2
        
    def _weight_variable(self,shape):
        #Set to 2 for testing (makes it easy to verify result using hand calculations)
        initial = tf.mul(tf.ones(shape),2)    #tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)
    
    def residualBlock(self, input, dialation, name):
        dialatedConvSig = tf.nn.sigmoid(self._cust_atrous_conv2d(input, dialation, name))
        dialatedConvTan = tf.nn.tanh(self._cust_atrous_conv2d(input, dialation, name))
        combined = tf.mul(dialatedConvSig,dialatedConvTan)
        skipConnection = tf.nn.conv2d(combined, self._weight_variable([1,1,1,1]), strides=[1,1,1,1],padding = "SAME")
        residual = skipConnection + input
        #Skip connection is used for softmax prediction
        #Residual is the input to the next layer
        return [residual, skipConnection]
        
        
    def calculateOutput(self, skipConnections):
        #Make sure you are reducing across the proper dimensions
        activatedSum = tf.nn.relu(tf.reduce_sum(skipConnections, axis=0))
        print(activatedSum.get_shape())
        conv1 = tf.nn.conv2d(activatedSum, self._weight_variable([1,1,1,numFilters]), strides = [1,1,1,1], padding = "SAME")
        conv1Activated = tf.nn.relu(conv1)
        print(conv1Activated.get_shape())
        conv2 = tf.nn.conv2s(acticatedSum, self._weight_variable())
        
        
        
        
        
        





net = Net(SAMPLESIZE)

#First causal (non dialeted) layer. Does not participate in residual block
#l1 = tf.nn.relu(net._cust_atrous_conv2d(net.input, dialation=1,name="name"))
#All other layers consist of residual blocks (except output)
#l2Resid, l2Skip = net.residualBlock(l1, dialation=1, name="l2")


output = net._cust_atrous_conv2d(net.input, dialation=1,name="name")



with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    result = sess.run(output, feed_dict={net.input: np.array([i for i in range(SAMPLESIZE)]).reshape((1,1,SAMPLESIZE,1))})
    print(result)
        