import tensorflow as tf
import numpy as np
import Audio

SAMPLESIZE = 5
NUMOUTPUTS = 256
RESULT_LENGTH = 5



class Net():
    def __init__(self, sampleSize):
        #height is added to accomodate tensorflows lack of support for 1d atrous convolutions 
        #height = 1
        numChannels = 1 
        self.sampleSize = sampleSize
        self.input = tf.placeholder(tf.float32, shape = [None,1, self.sampleSize, numChannels])
        self.skips = []
        self.prevLayer = self.input
        self.learningRate = 0.01

    def _cust_atrous_conv2d(self, input, dialation,numFilters, name):
        with tf.variable_scope("atrous_"+str(name)) as scope:
            const = tf.constant([0.0 for i in range(dialation)])
            inputShape = int(input.get_shape()[-1]) * int(input.get_shape()[-2]) 
            reshapedInput = tf.reshape(input,(inputShape,))
            concated = tf.slice(tf.concat(0,[const, reshapedInput]), [0],[self.sampleSize])
            stacked = tf.concat(3,[tf.reshape(concated,shape=(-1,1,self.sampleSize, 1)), input])
            stacked = tf.concat(3, [stacked for i in range(int(inputShape))])
            _,_,_,prevChannels = stacked.get_shape()
            output = tf.nn.conv2d(stacked, self._weight_variable([1,1,int(prevChannels),numFilters]), strides=[1,1,1,1], padding = "SAME")
            return output
        
    def _weight_variable(self,shape):
        #Set to 2 for testing (makes it easy to verify result using hand calculations)
        initial = tf.truncated_normal(shape, stddev=0.1) #tf.mul(tf.ones(shape),2)    #
        return tf.Variable(initial)
    
    def _residualBlock(self, input, dialation,numFilters, name):
        dialatedConvSig = tf.nn.sigmoid(self._cust_atrous_conv2d(input, dialation,numFilters, name))
        dialatedConvTan = tf.nn.tanh(self._cust_atrous_conv2d(input, dialation,numFilters, name))
        combined = tf.mul(dialatedConvSig,dialatedConvTan)
        print(combined)
        skipConnection = tf.nn.conv2d(combined, self._weight_variable([1,1,int(combined.get_shape()[-1]),numFilters]), strides=[1,1,1,1],padding = "SAME")    
        residual = skipConnection + input
        self.skips.append(skipConnection)
        return [residual, skipConnection]
        
    def _calculateOutput(self, skipConnections,numFilters):
        activatedSum = tf.nn.relu(tf.reduce_sum(skipConnections, axis=0))
        conv1 = tf.nn.conv2d(activatedSum, self._weight_variable([1,1,int(activatedSum.get_shape()[-1]),numFilters]), strides = [1,1,1,1], padding = "SAME")
        conv1Activated = tf.nn.relu(conv1)
        conv2 = tf.nn.conv2d(conv1Activated, self._weight_variable([1,1,int(conv1Activated.get_shape()[-1]), numFilters]), strides = [1,1,1,1], padding = "SAME")
        fc = tf.reshape(conv2, [SAMPLESIZE, -1])
        
        
        weights = self._weight_variable([int(fc.get_shape()[-1]),NUMOUTPUTS])
        soft =  tf.nn.softmax(tf.matmul(fc, weights))
        print("SOFT SHAPE:")
        print(soft.get_shape())
        return soft
        
        
    def _addLayer(self, prevLayer,dialation):
        self.prevLayer, _ = net._residualBlock(prevLayer, dialation=dialation,numFilters = 128, name="l2")
        
        
    def build(self, layers):
        #First causal (non dialeted) layer. Does not participate in residual block
        self.prevLayer = tf.nn.relu(net._cust_atrous_conv2d(self.prevLayer, dialation=1,numFilters=128,name="name"))
        for i in range(layers):
            #All other layers consist of residual blocks (except output)
            self._addLayer(self.prevLayer,2**i)
        #Generate output layer
        output = net._calculateOutput(self.skips, 128)
        print("OUTPUT SHAPE:")
        print(output.get_shape())
        return output
    
    def _getLoss(self, output, answers):
        print("___________________________")
        print(output.get_shape())
        print(answers.get_shape())
        print("___________________________")
        return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=answers,logits=output))
    
    def getUpdateOp(self, output, answers):
        loss = self._getLoss(output, answers)    
        optimizer = tf.train.AdamOptimizer(self.learningRate)
        grads = optimizer.compute_gradients(loss, tf.trainable_variables())
        return [optimizer.apply_gradients(grads),loss] #[optimizer.apply_gradients(grads), loss, grads]
        
        
        

net = Net(SAMPLESIZE)
output = net.build(5)
answerPH = tf.placeholder(tf.int32, [SAMPLESIZE])
lossOp, loss = net.getUpdateOp(output, answerPH)





with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(151):

        result = sess.run(lossOp, feed_dict={net.input:np.array(Audio.getSample()[0][:SAMPLESIZE]).reshape(1,1,SAMPLESIZE,1), answerPH:np.array(Audio.getSample()[0][1:(SAMPLESIZE+1)]).reshape(SAMPLESIZE)})
        if(i%10 ==0):
            
            error = sess.run(loss, feed_dict={net.input:np.array(Audio.getSample()[0][:SAMPLESIZE]).reshape(1,1,SAMPLESIZE,1), answerPH:np.array(Audio.getSample()[0][1:(SAMPLESIZE+1)]).reshape(SAMPLESIZE)})
            print("Error : %s | Iteration %s"%(error, i))
    
    result = sess.run(output, feed_dict={net.input:np.array(Audio.getSample()[0][:SAMPLESIZE]).reshape(1,1,SAMPLESIZE,1)})
    for e, f in zip(result,np.array(Audio.getSample()[0][:SAMPLESIZE]).reshape(SAMPLESIZE) ):
        print(np.argmax(e,axis=0))
        print(f)
        print("+======+")
        
        

    
    
## Create a new net with persisted weights to predict on a different length sample
def zeroFill(prevPreds):
## Testing the result:
    #Temporary, to make sure its doing its job
    #every 2*training sequence, split it in half and save the result 
    if(testing):
        prevPreds = []
        for iter in range(RESULT_LENGTH):
            result = sess.run(output, feed_dict=np.concat(prevPreds, np.zeros(RESULT_LENGTH-(iter+1))))
            prevPreds = result[:iter]
            
        

    
    
    
        