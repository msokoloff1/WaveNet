import tensorflow as tf
import numpy as np

SAMPLESIZE = 6

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
            concated = tf.slice(tf.concat(0,[const, reshapedInput]), [0],[self.sampleSize])
            stacked = tf.concat(3,[tf.reshape(concated,shape=(-1,1,self.sampleSize, 1)), input])
            output = tf.nn.conv2d(stacked, self.weight_variable([1,1,2,1]), strides=[1,1,1,1], padding = "SAME")
            print(output.get_shape())
            return output
        
    def weight_variable(self,shape):
        #Set to 2 for testing (makes it easy to verify result using hand calculations)
        initial = tf.mul(tf.ones(shape),2)    #tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)
    
    def _residualBlock(self, input):
        dialatedConv = self._cust_atrous_conv2d





net = Net(SAMPLESIZE)

#First causal (non dialeted) layer. Does not participate in residual block
l1 = tf.nn.relu(net._cust_atrous_conv2d(a.input, dialation=1,name="name"))
#All other layers consist of residual blocks (except output)
l2 = 


output = a._cust_atrous_conv2d(r1, dialation=1,name="name")



with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    result = sess.run(output, feed_dict={a.input: np.array([i for i in range(SAMPLESIZE)]).reshape((1,1,SAMPLESIZE,1))})
    print(result)
        