import numpy as np
def sigmoid(x):
    """
    Compute the sigmoid function for the input here.

    Arguments:
    x -- A scalar or numpy array.

    Return:
    s -- sigmoid(x)
    """
    ### YOUR CODE HERE
    s = 1 / (1 + np.exp(-x))

    ### END YOUR CODE
    return s

def test_sigmoid():

    """

    Some simple tests to get you started.

    Warning: these are not exhaustive.

    """

    print ("Running basic tests...")

    x = np.array([[1, 2], [-1, -2]])

    f = sigmoid(x)

    print (f)

    f_ans = np.array([

        [0.73105858, 0.88079708],

        [0.26894142, 0.11920292]])

    assert np.allclose(f, f_ans, rtol=1e-05, atol=1e-06)

    print ("You should verify these results by hand!\n")
if __name__ == "__main__":
    test_sigmoid()
