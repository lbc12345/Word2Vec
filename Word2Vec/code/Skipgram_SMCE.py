import random
import numpy as np
from utils.treebank import StanfordSentiment
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import time
import glob
import random
import os.path as op
import pickle as pickle
from softmax import softmax
#########################################################################################
### These are the functions you need to complete. 				    
### However, you may read the documents of support functions blow first             
### and this will help you figure out what you are doing.			   
#########################################################################################
def softmaxCostAndGradient(v, u_index, outputVectors, dataset):
    """
    Softmax cost function for word2vec models
    Implement the cost and gradients for one predicted word vector
    and one target word vector as a building block for word2vec
    models, assuming the softmax prediction function and cross
    entropy loss.

    Arguments:
    v -- Center word's input vector (a numpy row array with dimention: 1 x dimVectors) 
    u_index -- integer, the index of the context word
    outputVectors -- "output" vectors (as rows) for all tokens (a numpy matrix with dimention: N/2 x dimVectors) 
    dataset -- needed for negative sampling, unused here.

    Return:
    cost_temp -- cross entropy cost for the current pair of training sample
    gradv_temp -- the gradient with respect to the center word's input vector
    gradu_temp -- the gradient with respect to all the other word vectors's output vector

    We will not provide starter code for this function, but feel
    free to reference the code you previously wrote for this
    assignment!
    """
    ### YOUR CODE HERE
    ### Cost
    context_vector = np.zeros(outputVectors.shape[0])
    context_vector[u_index]=1
    cost_temp = np.sum(-context_vector*np.log(softmax(np.dot(v,outputVectors.T))))
    z = []
    for row in outputVectors:
        z.append(np.dot(row, v))
    ### Gradient :
    ### Calculate the grad for predictions(1 x dimVectors):
    z_new = softmax(np.array(z))
    '''i = 0
    gradv_temp = 0
    for u in outputVectors:
        gradv_temp += np.multiply(z_new[i], u)
        i += 1
    gradv_temp -= outputVectors[u_index]'''
    gradv_temp = np.dot(outputVectors.T, z_new - context_vector)
    ### Calculate the grad for outputVectors (N/2 x dimVectors):
    gradu_temp = np.dot((z_new - context_vector).reshape(outputVectors.shape[0], 1), v.reshape(1, outputVectors.shape[1]))
    ### END YOUR CODE
    return cost_temp, gradv_temp, gradu_temp

def skipgram(centerword, C, contextWords, tokens, inputVectors, outputVectors, dataset, word2vecCostAndGradient):
    """ 
    Skip-gram model in word2vec
    Implement the skip-gram model in this function.

    Arguments:
    centerword -- The center word
    C -- Integer, context size
    contextWords -- List of no more than 2*C strings, the context words
    tokens -- A dictionary that maps words to their indices in
              the word vector list
    inputVectors -- "input" word vectors (as rows) for all tokens (a numpy matrix with dimention: N/2 x dimVectors) 
    outputVectors -- "output" word vectors (as rows) for all tokens (a numpy matrix with dimention: N/2 x dimVectors) 
    word2vecCostAndGradient -- The cost and gradient function.

    Return:
    cost -- the cost function value for the skip-gram model
    gradIn -- the gradient with respect to the word vectors
    gradOut -- the gradient with respect to the word vectors
    """
    cost = 0.0
    gradIn = np.zeros(inputVectors.shape)
    gradOut = np.zeros(outputVectors.shape)
    ### YOUR CODE HERE
    v_index = tokens[centerword]              
    v = inputVectors[v_index]          
    u_indexs = [tokens[word] for word in contextWords ]          
    for u_index in u_indexs:
        cost_temp, gradv_temp, gradu_temp = word2vecCostAndGradient(v, u_index, outputVectors, dataset)
        cost = cost + cost_temp
        gradIn[v_index] = gradIn[v_index] + gradv_temp
        gradOut = gradOut + gradu_temp
    ### END YOUR CODE

    return cost, gradIn, gradOut

#########################################################################################
### Don't modify the following functions. Otherwise, you may account strange problems.
### The recommented order of functions for you to read is:                      
### (1) run									  
### (2) sgd									     
### (3) word2vec_sgd_wrapper							     
### (4) skipgram 								      
### (5) softmaxCostAndGradient                                                       
#########################################################################################
def normalizeRows(x):
    """ 
    Row normalization function
    Implement a function that normalizes each row of a matrix to have unit length.
    """
    denom = np.apply_along_axis(lambda x: np.sqrt(x.T.dot(x)), 1, x)
    x /= denom[:, None]

    return x

def load_saved_params():
    """
    A helper function that loads previously saved parameters and resets iteration start.
    """
    st = 0
    for f in glob.glob("saved_params_*.pickle"):
        iter = int(op.splitext(op.basename(f))[0].split("_")[2])
        if (iter > st):
            st = iter

    if st > 0:
        with open("saved_params_%d.pickle" % st, "rb") as f:
            params = pickle.load(f)
            state = pickle.load(f)
        return st, params, state
    else:
        return st, None, None

def save_params(iter, params):
    with open("saved_params_%d.pickle" % iter, "wb") as f:
        pickle.dump(params, f)
        pickle.dump(random.getstate(), f)

def Reshape(x): 
    if len(x.shape) == 0:
        x = np.array([[x]])
    elif len(x.shape) == 1:
        x = np.array([x])
    elif x.shape[0] > 1 and x.shape[1] == 1:
        x = x.T
    return x

def gradcheck_naive(f, x):
    """ Gradient check for a function f.

    Arguments:
    f -- a function that takes a single argument and outputs the
         cost and its gradients
    x -- the point (numpy array) to check the gradient at
    """
    rndstate = random.getstate()
    random.setstate(rndstate)
    fx, grad = f(x) # Evaluate function value at original point
    h = 1e-4        # Do not change this!
    # Iterate over all indexes in x
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index
        x[ix] = x[ix] + h
        random.setstate(rndstate)
        f_temp_plus, T = f(x)
        f_temp_plus = Reshape(f_temp_plus)
        x[ix] = x[ix] - 2*h
        random.setstate(rndstate)
        f_temp_minus, T = f(x)
        f_temp_minus = Reshape(f_temp_minus)
        x[ix] = x[ix] + h
        if f_temp_plus.shape[1] == 1:                              
            numgrad = (f_temp_plus - f_temp_minus) / ( 2 * h )
        elif len(fx.shape) == 1:                                   
            numgrad = (f_temp_plus[0][ix[0]] - f_temp_minus[0][ix[0]]) / ( 2 * h )
        elif fx.shape[0] > 1 and fx.shape[1] == 1:               
            numgrad = (f_temp_plus.T[ix] - f_temp_minus.T[ix]) / (2 * h)
        else:                                                    
            numgrad = (f_temp_plus[ix] - f_temp_minus[ix]) / (2 * h)
        reldiff = abs(numgrad - grad[ix]) / max(1, abs(numgrad), abs(grad[ix]))
        if reldiff > 1e-5:
            print ("Gradient check failed.")
            print ("First gradient error found at index %s" % str(ix))
            print ("Your gradient: %f \t Numerical gradient: %f" % (grad[ix], numgrad))
            assert 1 == 2
            return
        it.iternext() # Step to next dimension
    print ("Gradient check passed!")

def word2vec_sgd_wrapper(word2vecModel, tokens, wordVectors, dataset, C,  word2vecCostAndGradient):
    """ 
    Function: Calculate the summation of cost and gradient in each iter. i.e. in each batch.

    Arguments:
    word2vecModel -- The type of model.

         Specificly, it's:
	 "skipgram"         

    tokens -- The index of all words.

         Specificly, it's:
         "dataset.tokens()"

    wordVectors -- The input vectors and output vectors.

         Specificly, it's:
         "x"

    dataset -- The dictionary of all words and their corresponding index .

	 Specificly, it's:
         "StanfordSentiment()"

    C -- Window size.
         
         Specificly, it's:
         "5"
    
    word2vecCostAndGradient -- The type of cost function.
         
         Specificly, it's:
         "softmaxCostAndGradient"

    Return:
    cost -- the parameter value after SGD finishes.
    grad -- the gradient of all input and output vectors (derived from samples in this batch).
    """
    ### In this iter, we sample 50 windows randomly and calculate the average cost and average gradient of all windows
    batchsize = 50
    ### Initialize the cost to be 0
    cost = 0.0
    ### Initialize the gradient of all input and output vectors to be 0
    grad = np.zeros(wordVectors.shape) 
    ### Record the twice number of all vocabularies                   
    N = wordVectors.shape[0]
    ### Get all the input vectors (a numpy matrix with dimention: N/2 x dimVectors)                             
    inputVectors = wordVectors[:int(N / 2), :]    
    ### Get all the output vectors (a numpy matrix with dimention: N/2 x dimVectors)          
    outputVectors = wordVectors[int(N / 2):, :] 
    ### For each word window, we call the            
    for i in range(batchsize):    
        ### Pick up a word window randomly with length less than 5*2+1=11.                          
        C1 = random.randint(1, C)
        ### Find out the specific word in this word window. i.e. "centerword" and "context" are both charts                         
        centerword, context = dataset.getRandomContext(C1) 
        if word2vecModel == skipgram:
            denom = 1
        else:
            denom = 1 
        ### In skipgram model, calculate the cost and gradient of each word window.
        c, gin, gout = word2vecModel(centerword, C1, context, tokens, inputVectors, outputVectors, dataset, word2vecCostAndGradient)
        ### Calculate the average cost and average average gradient of all windows
        cost += c / batchsize / denom
        grad[:int(N / 2), :] += gin / batchsize  / denom 
        grad[int(N / 2):, :] += gout / batchsize  / denom
    return cost, grad

def sgd(f, x0, step, iterations, postprocessing=None, useSaved=False,PRINT_EVERY=10):
    """ 
    Stochastic Gradient Descent:
    Implement the stochastic gradient descent method in this function.
    Arguments:
    f -- the function to optimize, it should take a single
         argument and yield two outputs, a cost and the gradient
         with respect to the arguments

         Specificly, it's a lambda function:
	 "word2vec_sgd_wapper(skipgram, tokens, vec, dataset, C, softmaxCostAndGradient)"         
	 
         The variable is:
         "vec"

    x0-- the initial point to start SGD from

         Specificly, it's:
         "wordVectors"

    step -- the step size for SGD

         Specificly, it's:
         "0.3"

    iterations -- total iterations to run SGD for

	 Specificly, it's:
         "40000"

    postprocessing -- postprocessing function for the parameters
                      if necessary. In the case of word2vec we will need to
                      normalize the word vectors to have unit length.
         
         Specificly, it's:
         "None"
    
    useSaved -- Whether to use the embeddings saved during the training process
         
         Specificly, it's:
         "True"

    PRINT_EVERY -- specifies how many iterations to output loss
         
         Specificly, it's:
         "10"

    Return:
    x -- the parameter value after SGD finishes
    """
    ### Anneal learning rate every several iterations
    ANNEAL_EVERY = 20000
    SAVE_PARAMS_EVERY = 5000
    if useSaved:
        start_iter, oldx, state = load_saved_params()
        if start_iter > 0:
            x0 = oldx
            step *= 0.5 ** (start_iter / ANNEAL_EVERY)
        if state:
            random.setstate(state)
    else:
        start_iter = 0

    ### x = [input vectors; outputvectors] (concatenated)
    x = x0

    ### 
    if not postprocessing:
        postprocessing = lambda x: x
    expcost = None

    ### We will train this model(skipgram+softmaxCostAndGradient) for 40000 times
    for iter in range(start_iter + 1, iterations + 1):
        cost = None
        ### Call "word2vec_sgd_wapper" function, return with the current average cost 
        ### of a word window and the gradient for all input vectors and output vectors.
        cost, grad = f(x)
        ### Update all input vectors and output vectors.
        x -= step * grad
        ### Conduct some postprocessing to all input vectors and output vectors.
        postprocessing(x)
        ### Print emperical cost every 10 iters.
        if iter % PRINT_EVERY == 0:
            if not expcost:
                expcost = cost
            else:
                expcost = .95 * expcost + .05 * cost
            print ("iter %d: %f" % (iter, expcost))
        ### Save the intermedia results every 5000 iters.
        if iter % SAVE_PARAMS_EVERY == 0 and useSaved:
            save_params(iter, x)
        ### After 20000 iters, we set the step size to half.
        if iter % ANNEAL_EVERY == 0:
            step *= 0.5
    return x

def run():
    ### Here is the main body of this file. We initialize the model and clean up the dataset
    ### Reset the random seed to make sure that everyone gets the same results
    random.seed(314)
    dataset = StanfordSentiment()
    tokens = dataset.tokens()
    nWords = len(tokens)

    ### We are going to train 10-dimensional vectors for this assignment
    dimVectors = 10

    ### The maximum half context size
    C = 5

    ### Reset the random seed to make sure that everyone gets the same results
    random.seed(31415)
    np.random.seed(9265)

    ### Start the clock when we begin to train this model
    startTime=time.time()

    ### The initial point to start SGD from
    wordVectors = np.concatenate(
    ((np.random.rand(nWords, dimVectors) - 0.5) /
       dimVectors, np.zeros((nWords, dimVectors))),
    axis=0)

    ### Call the sgd function to train our model, 
    wordVectors = sgd(lambda vec: word2vec_sgd_wrapper(skipgram, tokens, vec, dataset, C, softmaxCostAndGradient), wordVectors, 0.3, 40000, None, True, PRINT_EVERY=10)

    ### Note that normalization is not called here. This is not a bug,
    ### normalizing during training loses the notion of length.

    print ("sanity check: cost at convergence should be around or below 10")
    print ("training took %d seconds" % (time.time() - startTime))

    ### Concatenate the input and output word vectors
    wordVectors = np.concatenate(
    (wordVectors[:nWords,:], wordVectors[nWords:,:]),
    axis=0)
    ### wordVectors = wordVectors[:nWords,:] + wordVectors[nWords:,:]

    ### Visualize word embeddings
    visualizeWords = [
    "the", "a", "an", ",", ".", "?", "!", "``", "''", "--",
    "good", "great", "cool", "brilliant", "wonderful", "well", "amazing",
    "worth", "sweet", "enjoyable", "boring", "bad", "waste", "dumb",
    "annoying"]

    visualizeIdx = [tokens[word] for word in visualizeWords]
    visualizeVecs = wordVectors[visualizeIdx, :]
    temp = (visualizeVecs - np.mean(visualizeVecs, axis=0))
    covariance = 1.0 / len(visualizeIdx) * temp.T.dot(temp)
    U,S,V = np.linalg.svd(covariance)
    coord = temp.dot(U[:,0:2])

    for i in range(len(visualizeWords)):
        plt.text(coord[i,0], coord[i,1], visualizeWords[i],
                 bbox=dict(facecolor='green', alpha=0.1))

    plt.xlim((np.min(coord[:,0]), np.max(coord[:,0])))
    plt.ylim((np.min(coord[:,1]), np.max(coord[:,1])))

    plt.savefig('q3_word_vectors.png')

def test_word2vec():
    """ Interface to the dataset for negative sampling """
    dataset = type('dummy', (), {})()
    def dummySampleTokenIdx():
        return random.randint(0, 4)
    def getRandomContext(C):
        tokens = ["a", "b", "c", "d", "e"]
        return tokens[random.randint(0, 4)], \
               [tokens[random.randint(0, 4)] for i in range(2 * C)]
    dataset.sampleTokenIdx = dummySampleTokenIdx
    dataset.getRandomContext = getRandomContext
    random.seed(31415)
    np.random.seed(9265)
    dummy_vectors = normalizeRows(np.random.randn(10, 3))
    dummy_tokens = dict([("a", 0), ("b", 1), ("c", 2), ("d", 3), ("e", 4)])
    print ("==== Gradient check for skip-gram ====")
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(skipgram, dummy_tokens, vec, dataset, 5, softmaxCostAndGradient), dummy_vectors)


if __name__ == "__main__":
    test_word2vec()
    run()
    


















































































