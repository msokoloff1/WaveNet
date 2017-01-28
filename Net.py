import tensorflow as tf
import numpy as np
import Audio
from random import randint
import time

SAMPLESIZE = 25000
NUMOUTPUTS = 256
RESULT_LENGTH = 100000



class Net():
    def __init__(self, sampleSize):
        numChannels = 1 
        #self.sampleSize = sampleSize
        
                                                          #Variable sequence length
        self.input = tf.placeholder(tf.float32, shape = [1, None, numChannels])
        print(type(self.input))
        self.skips = []
        self.prevLayer = self.input
        self.learningRate = 0.001
        self.layerss = []
        self.once = False
        
        
        
    def _cust_atrous_conv2d(self, input, dialation,numFilters, name):
        with tf.variable_scope("atrous_"+str(name)) as scope:
            print(input.get_shape())
            if(not self.once):
                inputFilters = 1
                self.once = True
            else:
                inputFilters = 128   
                
            out = self.causal_conv(input, self._weight_variable([2, inputFilters, numFilters]), dialation)
            return out
        
    def time_to_batch(self,value, dilation, name='time_to_batch'):
        with tf.name_scope(name):
            shape = tf.shape(value)
            pad_elements = dilation - 1 - (shape[1] + dilation - 1) % dilation
            padded = tf.pad(value, [[0, 0], [0, pad_elements], [0, 0]])
            reshaped = tf.reshape(padded, [-1, dilation, shape[2]])
            transposed = tf.transpose(reshaped, perm=[1, 0, 2])
            return tf.reshape(transposed, [shape[0] * dilation, -1, shape[2]])

    def batch_to_time(self,value, dilation, name='batch_to_time'):
        with tf.name_scope(name):
            shape = tf.shape(value)
            prepared = tf.reshape(value, [dilation, -1, shape[2]])
            transposed = tf.transpose(prepared, perm=[1, 0, 2])
            return tf.reshape(transposed,
                              [tf.div(shape[0], dilation), -1, shape[2]])
        
    def causal_conv(self,value, filter_, dilation, name='causal_conv'):
        with tf.name_scope(name):

            padding = [[0, 0], [dilation, 0], [0, 0]]
            padded = tf.pad(value, padding)
            if dilation > 1:
                transformed = self.time_to_batch(padded, dilation)
                conv = tf.nn.conv1d(transformed, filter_, stride=1, padding='SAME')
                restored = self.batch_to_time(conv, dilation)
            else:
                restored = tf.nn.conv1d(padded, filter_, stride=1, padding='SAME')
                
            result = tf.slice(restored,
                              [0, 0, 0],
                              [-1, tf.shape(value)[1], -1])
            return result
      
        
    def _weight_variable(self,shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)
    
    def _residualBlock(self, input, dialation,numFilters, name='_residualBlock'):
        with tf.name_scope(name):
            dialatedConvSig = tf.nn.sigmoid(self._cust_atrous_conv2d(input, dialation,numFilters, name))
            dialatedConvTan = tf.nn.tanh(self._cust_atrous_conv2d(input, dialation,numFilters, name))
            combined = tf.mul(dialatedConvSig,dialatedConvTan)
            skipConnection     = tf.nn.conv1d(combined, self._weight_variable([1,128,numFilters]), stride=1,padding = "SAME")
            residualConnection = tf.nn.conv1d(combined, self._weight_variable([1,128,numFilters]), stride=1, padding = "SAME")     
            residual = residualConnection + input
            self.skips.append(skipConnection)
            return [residual, skipConnection]
        
    def _calculateOutput(self, skipConnections,numFilters, name='_calcOutut'):
        with tf.name_scope(name):
            activatedSum = tf.nn.relu(tf.reduce_sum(skipConnections, axis=0))
            conv1 = tf.nn.conv1d(activatedSum, self._weight_variable([1,128,numFilters]), stride = 1, padding = "SAME")
            conv1Activated = tf.nn.relu(conv1)
            conv2 = tf.nn.conv1d(conv1Activated, self._weight_variable([1,128, NUMOUTPUTS]), stride = 1, padding = "SAME")
            out = tf.reshape(conv2, [-1,256])
            #fc = tf.reshape(conv2, [SAMPLESIZE, -1])
            
            #weights = self._weight_variable([128,NUMOUTPUTS])
            #out =  tf.matmul(fc,weights)#tf.nn.softmax(tf.matmul(fc, weights))
            
            return out
        
         
    def _addLayer(self, prevLayer,dialation, numfilters):
        self.prevLayer, _ = net._residualBlock(prevLayer, dialation=dialation,numFilters = numfilters, name="l2")
        
        
    def build(self, layers):
        with tf.variable_scope("trainNet") as scope:
            #First causal (non dialeted) layer. Does not participate in residual block
            self.prevLayer = tf.nn.relu(net._cust_atrous_conv2d(self.prevLayer, dialation=1,numFilters=128,name="name"))
            for i in range(layers):
                #All other layers consist of residual blocks (except output)
                self._addLayer(self.prevLayer,2**i, 128)
            #Generate output layer
            output = net._calculateOutput(self.skips, 128)
            return output
    
    def _getLoss(self, output, answers):
        return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=answers,logits=output))
    
    def getUpdateOp(self, output, answers):
        loss = self._getLoss(output, answers)    
        optimizer = tf.train.AdamOptimizer(self.learningRate)
        grads = optimizer.compute_gradients(loss, tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='trainNet'))
        return [optimizer.apply_gradients(grads),loss] #[optimizer.apply_gradients(grads), loss, grads]
        
        

    
net = Net(SAMPLESIZE)
output = net.build(13)
answerPH = tf.placeholder(tf.int32, [SAMPLESIZE])
lossOp, loss = net.getUpdateOp(output, answerPH)





with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sample = Audio.getSample()
    for i in range(2001):

        result = sess.run(lossOp, feed_dict={net.input:np.array(Audio.getSample()[0][:SAMPLESIZE]).reshape(1,SAMPLESIZE,1), answerPH:np.array(sample[0][1:(SAMPLESIZE+1)]).reshape(SAMPLESIZE)})
        #result = sess.run(lossOp, feed_dict={net.input:np.array(Audio.getSample()[0][1000:(SAMPLESIZE+1000)]).reshape(1,SAMPLESIZE,1), answerPH:np.array(Audio.getSample()[0][1001:(SAMPLESIZE+1001)]).reshape(SAMPLESIZE)})
        #result = sess.run(lossOp, feed_dict={net.input:np.array(Audio.getSample()[0][2000:(SAMPLESIZE+2000)]).reshape(1,SAMPLESIZE,1), answerPH:np.array(Audio.getSample()[0][2001:(SAMPLESIZE+2001)]).reshape(SAMPLESIZE)})
        #result = sess.run(lossOp, feed_dict={net.input:np.array(Audio.getSample()[0][3000:(SAMPLESIZE+3000)]).reshape(1,SAMPLESIZE,1), answerPH:np.array(Audio.getSample()[0][3001:(SAMPLESIZE+3001)]).reshape(SAMPLESIZE)})
        if(i%10 ==0):
            
            error = sess.run(loss, feed_dict={net.input:np.array(sample[0][:SAMPLESIZE]).reshape(1,SAMPLESIZE,1), answerPH:np.array(sample[0][1:(SAMPLESIZE+1)]).reshape(SAMPLESIZE)})
            print("Error : %s | Iteration %s"%(error, i))
    
    result = sess.run(output, feed_dict={net.input:np.array(sample[0][:SAMPLESIZE]).reshape(1,SAMPLESIZE,1)})
    print("RESULT SHAPE:")
    print(result.shape)
    for e, f in zip(result,np.array(Audio.getSample()[0][1:(SAMPLESIZE+1)]).reshape(SAMPLESIZE) ):
        print(np.argmax(e,axis=0))
        print(f)
        print("+======+")
       
    print("EEEENNNNNDDDDD") 


        
        
    
    
    
    seed = randint(115,116)
    results = []
    
    for timeStep in range(RESULT_LENGTH):
        start = time.time()
        if(timeStep == 0):
            example = np.array(seed).reshape(1,-1,1)
        else:
            example = results[-SAMPLESIZE:]    
            example = np.array(example).reshape(1,-1,1)
            
        
            
        endPrep =  time.time() - start
        startRun = time.time()
        result = sess.run(output, feed_dict={net.input:example})
        results.append(np.argmax(result, axis = 1)[-1])
        
        endAll = time.time() - start
        endRun = time.time() - startRun 
        if(timeStep % 50 == 0 ):
            print("Step %s/%s | Time(all) %s | Time(prep) %s | Time(run) %s"%(timeStep,RESULT_LENGTH, endAll, endPrep, endRun))
        
        
    Audio.writeSample("testing123.wav",  np.array(results).reshape(-1))
    
    
    
    
    







    
    
    
        