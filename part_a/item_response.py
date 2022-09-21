from utils import *

import numpy as np
import scipy
from scipy import special
from numpy import random
import numpy.linalg as LA
from matplotlib import pyplot as plt

np.random.seed(16)


def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(matrix, theta, beta):
    """ Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param matrix: 2D sparse matrix
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################

    log_lklihood = 0.0
    matrix = matrix.todense()

    # This calculates student k - all questions
    z = theta - beta.T
    y = sigmoid(z)
    # Mask the NaN values to be avoided in calculations
    masked_matrix = np.ma.array(matrix, mask=np.isnan(matrix))

    log_lklihood = np.nansum(
        masked_matrix * np.log((y)) + (1 - masked_matrix) * np.log(1-y))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return -log_lklihood


def update_theta_beta(matrix, theta, beta, lr):
    """ Update theta and beta using gradient descent.

    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta

    You may optionally replace the function arguments to receive a matrix.

    :param matrix: 2D sparse matrix
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :return: tuple of vectors
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################

    z = theta - beta.T
    y = sigmoid(z)
    if scipy.sparse.issparse(matrix):
        c = np.ma.array(matrix.todense(), mask=np.isnan(matrix.todense()))
    else:
        c = np.ma.array(matrix, mask=np.isnan(matrix))
    t = theta - lr * np.nansum(-c + y, axis=1)
    b = beta - lr * np.nansum(c - y, axis=0).T

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return t, b


def irt(matrix, val_data, lr, iterations):
    """ Train IRT model.

    You may optionally replace the function arguments to receive a matrix.

    :param matrix: 2D sparse matrix
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, val_acc_lst)
    """
    # TODO: Initialize theta and beta.
    N, M = np.shape(matrix)

    theta = np.random.randn(N, 1)

    beta = np.random.randn(M, 1)

    val_acc_lst = []

    val_smat = build_sparse_mat(val_data)

    # Log likelihoods of training and validation data at each iterations
    train_llds = []
    val_llds = []

    for i in range(iterations):
        neg_lld = neg_log_likelihood(matrix, theta=theta, beta=beta)
        val_lld = neg_log_likelihood(val_smat, theta, beta)

        train_llds.append(neg_lld)
        val_llds.append(val_lld)

        score = evaluate(data=val_data, theta=theta, beta=beta)
        val_acc_lst.append(score)
        print("NLLK: {} \t Score: {}".format(neg_lld, score))

        if i % 2:
            theta = update_theta_beta(matrix, theta, beta, lr)[0]
        else:
            beta = update_theta_beta(matrix, theta, beta, lr)[1]

    iter_lst = np.arange(1, iterations + 1)

    # Plot the log likelihood of training and validation data
    # plt.plot(iter_lst, train_llds, label="Train log likelihood")
    # plt.plot(iter_lst, val_llds, label="Validation log likelihood")
    # plt.xlabel("Iterations")
    # plt.ylabel("Log Likelihood")
    # plt.legend()
    # plt.savefig("trainvsval")
    # plt.show()

    # TODO: You may change the return values to achieve what you want.
    return theta, beta, val_acc_lst


def evaluate(data, theta, beta):
    """ Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) \
        / len(data["is_correct"])




def main():
    train_data = load_train_csv("../data")
    # You may optionally use the sparse matrix.
    sparse_matrix = load_train_sparse("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    #####################################################################
    # TODO:                                                             #
    # Tune learning rate and number of iterations. With the implemented #
    # code, report the validation and test accuracy.                    #
    #####################################################################

    hyperparameters = {
        "lr": 0.01,
        "iterations": 10
    }
    theta, beta, val_accList = irt(sparse_matrix, val_data,
                                   hyperparameters["lr"], hyperparameters["iterations"])

    test_acc = evaluate(test_data, theta, beta)
    print("Test accuracy: {}".format(test_acc))

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # TODO:                                                             #
    # Implement part (d)                                                #
    #####################################################################

    # List of indexes of questions we selected
    question_index = [0, 1, 2]
    j_one = []
    j_two = []
    j_three = []

    # List of theta values to range over
    theta_list = list(np.arange(0, 1.1, 0.1))

    for theta_val in theta_list:
        # For each question, get its probability of being correctly answered based on a number of theta values
        j_one.append(sigmoid(theta_val - beta[question_index[0], 0]))
        j_two.append(sigmoid(theta_val - beta[question_index[1], 0]))
        j_three.append(sigmoid(theta_val - beta[question_index[2], 0]))

    plt.plot(theta_list, j_one, label="j_1")
    plt.plot(theta_list, j_two, label="j_2")
    plt.plot(theta_list, j_three, label="j_3")
    plt.xlabel("Theta values")
    plt.ylabel("P(c_ij)")
    plt.legend()
    plt.savefig("partd")
    plt.show()
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
