import numpy as np
def softmax(x):
    """
    Function: Compute the softmax function for each row of the input x.
    
    Tips:  (1)It is crucial that this function is optimized for speed because it will be used 
              frequently in later code. You might find numpy functions np.exp, np.sum,
              np.reshape, np.max, and numpy broadcasting useful for this task.

	      Numpy broadcasting documentation:
      	      http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html

	   (2)You should also make sure that your code works for a single D-dimention vector 
	      (treat the vector as a single row) and for N x D matrices. This may be useful for
              testing later.

	   (3)Also, make sure that the dimensions of the output match the input.

    Arguments:
    x -- A D dimensional vector or N X D dimensional numpy matrix.

    Return:
    x -- You are allowed to modify x in-place
    """
    orig_shape = x.shape
    if len(x.shape) > 1:
       ### Here is your implementation when x is a matrix. 
       ### Compute softmax function for each row of x.
       ### YOUR CODE HERE
        i = 0
        y = []
        for row in x:
            total = np.sum(np.exp(np.subtract(row, np.max(row))))
            row = np.divide(np.exp(np.subtract(row, np.max(row))), total)
            y.append(row)
        z = np.array(y)
        x = z
            


       ### END YOUR CODE
    else:
       ### Here is your implementation when x is a vector.
       ### YOUR CODE HERE
        total = np.sum(np.exp(np.subtract(x, np.max(x))))
        x = np.divide(np.exp(np.subtract(x, np.max(x))), total)

       ### END YOUR CODE

    assert x.shape == orig_shape
    return x

def test_softmax():
    """
    Function: Some simple tests to get you started. You may want to add more tests.
              When there is no warning or error reported, you pass these tests.
    
    Tips:  (1)Whether your code works for a single D-dimention vector and for N x D matrices?

	   (2)Whether the dimensions of the output match the input?

    Arguments:
    None

    Return:
    None
    """
    print( "Running basic tests..." )
    test1 = softmax( np.array( [1, 2] ) )
    print( test1 )
    ans1 = np.array( [0.26894142, 0.73105858] )
    assert np.allclose( test1, ans1, rtol = 1e-05, atol = 1e-06 )

    test2 = softmax( np.array( [[1001, 1002], [3, 4]] ) )
    print( test2 )
    ans2 = np.array( [[0.26894142, 0.73105858],
		      [0.26894142, 0.73105858]] )
    assert np.allclose( test2, ans2, rtol = 1e-05, atol = 1e-06 )

    test3 = softmax( np.array( [-1001, -1002] ) )
    print( test3 )
    ans3 = np.array( [0.73105858, 0.26894142] )
    assert np.allclose( test1, ans1, rtol = 1e-05, atol = 1e-06 )

if __name__ == "__main__":
    test_softmax()















