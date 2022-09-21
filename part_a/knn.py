from sklearn.impute import KNNImputer
from utils import *
from matplotlib import pyplot as plt


def knn_impute_by_user(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    student similarity. Return the accuracy on valid_data.

    See https://scikit-learn.org/stable/modules/generated/sklearn.
    impute.KNNImputer.html for details.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(matrix)
    acc = sparse_matrix_evaluate(valid_data, mat)
    print("Validation Accuracy by user: {}".format(acc))
    return acc


def knn_impute_by_item(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    question similarity. Return the accuracy on valid_data.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float, (return the matrix for ensemble predictions)
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    nbrs = KNNImputer(n_neighbors=k)
    mat = nbrs.fit_transform(matrix.T)
    acc = sparse_matrix_evaluate(valid_data, mat.T)
    print("Validation Accuracy by item: {}".format(acc))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return acc,mat


def main():
    sparse_matrix = load_train_sparse("../data").toarray()
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    #####################################################################
    # TODO:                                                             #
    # Compute the validation accuracy for each k. Then pick k* with     #
    # the best performance and report the test accuracy with the        #
    # chosen k*.                                                        #
    #####################################################################

    # Check accuracy based on different k values
    k_array = [1, 6, 11, 16, 21, 26]
    user_acc_array = []
    item_acc_array = []

    curr_user_acc = 0
    curr_item_acc = 0

    # Get k* for both user and item based filtering
    for k in k_array:
        user_acc = knn_impute_by_user(sparse_matrix, val_data, k)
        user_acc_array.append(user_acc)
        if user_acc > curr_user_acc:
            user_k_star = k
            curr_user_acc = user_acc

        item_acc = knn_impute_by_item(sparse_matrix, val_data, k)[0]
        item_acc_array.append(item_acc)
        if item_acc > curr_item_acc:
            item_k_star = k
            curr_item_acc = item_acc

    # Plot accuracy on validation data as a function of k for both filtering
    # plt.plot(k_array, user_acc_array)
    # plt.savefig("user-acc")
    # plt.show()

    # plt.plot(k_array, item_acc_array)
    # plt.savefig("item-acc")
    # plt.show()

    print(
        "k* for user: {0}, k* for item: {1}".format(user_k_star, item_k_star))

    print("User-based test accuracy: ")
    user_test_acc = knn_impute_by_user(sparse_matrix, test_data, user_k_star)

    print("Item-based test accuracy: ")
    item_test_acc = knn_impute_by_item(sparse_matrix, test_data, item_k_star)[0]

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
