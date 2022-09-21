from utils import *
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

import numpy as np
import torch


def load_data(base_path="../data"):
    """ Load the data in PyTorch Tensor.

    :return: (zero_train_matrix, train_data, valid_data, test_data)
        WHERE:
        zero_train_matrix: 2D sparse matrix where missing entries are
        filled with 0.
        train_data: 2D sparse matrix
        valid_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
        test_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
    """
    train_matrix = load_train_sparse(base_path).toarray()
    valid_data = load_valid_csv(base_path)
    test_data = load_public_test_csv(base_path)

    zero_train_matrix = train_matrix.copy()
    # Fill in the missing entries to 0.
    zero_train_matrix[np.isnan(train_matrix)] = 0
    # Change to Float Tensor for PyTorch.
    zero_train_matrix = torch.FloatTensor(zero_train_matrix)
    train_matrix = torch.FloatTensor(train_matrix)

    return zero_train_matrix, train_matrix, valid_data, test_data


class AutoEncoder(nn.Module):
    def __init__(self, num_question, k=100):
        """ Initialize a class AutoEncoder.

        :param num_question: int
        :param k: int
        """
        super(AutoEncoder, self).__init__()

        # Define linear functions.
        self.g = nn.Linear(num_question, k)
        self.h = nn.Linear(k, num_question)
        self.activation = nn.Sigmoid()

    def get_weight_norm(self):
        """ Return ||W^1||^2 + ||W^2||^2.

        :return: float
        """
        g_w_norm = torch.norm(self.g.weight, 2) ** 2
        h_w_norm = torch.norm(self.h.weight, 2) ** 2
        return g_w_norm + h_w_norm

    def forward(self, inputs):
        """ Return a forward pass given inputs.

        :param inputs: user vector.
        :return: user vector.
        """
        #####################################################################
        # Implement the function as described in the docstring.             #
        # Use sigmoid activations for f and g.                              #
        #####################################################################      
        encoded = self.activation(self.g(inputs))
        decoded = self.activation(self.h(encoded))

        #####################################################################
        #                       END OF YOUR CODE                            #
        #####################################################################
        return decoded


def train(model, lr, lamb, train_data, zero_train_data, valid_data, num_epoch):
    """ Train the neural network, where the objective also includes
    a regularizer.

    :param model: Module
    :param lr: float
    :param lamb: float
    :param train_data: 2D FloatTensor
    :param zero_train_data: 2D FloatTensor
    :param valid_data: Dict
    :param num_epoch: int
    :return: None
    """ 
    
    # Tell PyTorch you are training the model.
    model.train()

    # Define optimizers and loss function.
    optimizer = optim.SGD(model.parameters(), lr=lr)
    num_student = train_data.shape[0]

    for epoch in range(0, num_epoch):
        train_loss = 0.

        for user_id in range(num_student):
            inputs = Variable(zero_train_data[user_id]).unsqueeze(0)
            target = inputs.clone()

            optimizer.zero_grad()
            output = model(inputs)

            # Mask the target to only compute the gradient of valid entries.
            nan_mask = np.isnan(train_data[user_id].unsqueeze(0).numpy())
            target[0][nan_mask] = output[0][nan_mask]

            loss = torch.sum((output - target) ** 2.) + ((lamb / 2) * (model.get_weight_norm()))
            loss.backward()

            train_loss += loss.item()
            optimizer.step()

        valid_acc = evaluate(model, zero_train_data, valid_data)[0]
        print("Epoch: {} \tTraining Cost: {:.6f}\t "
              "Valid Acc: {}".format(epoch, train_loss, valid_acc))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


def evaluate(model, train_data, valid_data):
    """ Evaluate the valid_data on the current model.

    :param model: Module
    :param train_data: 2D FloatTensor
    :param valid_data: A dictionary {user_id: list,
    question_id: list, is_correct: list}
    :return: float, list of predictions
    """
    # Tell PyTorch you are evaluating the model.
    model.eval()

    total = 0
    correct = 0
    predictions = []
    for i, u in enumerate(valid_data["user_id"]):
        inputs = Variable(train_data[u]).unsqueeze(0)
        output = model(inputs)

        guess = output[0][valid_data["question_id"][i]].item() >= 0.5
        predictions.append(float(guess))
        if guess == valid_data["is_correct"][i]:
            correct += 1
        total += 1
    return correct / float(total), predictions


def main():
    zero_train_matrix, train_matrix, valid_data, test_data = load_data()
    #####################################################################
    # Try out 5 different k and select the best k using the             #
    # validation set.                                                   #
    #####################################################################

    # Set optimization hyperparameters.
    num_epoch = 33

    # Set model hyperparameters.
    num_qs = list(train_matrix.shape)[1]  # num of columns in train_matrix

    ###########################
    ### Uncomment below code to run through hyperparameters combos
    ### Print results to .txt file and run thru nn_help.py to find optimals
    ### Running thru below code can take up to 30 minutes to run. 

    # for k in [10, 50, 100, 200, 500]: 
    #     for lr in [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]:
    #         # for _lambda in [0.001, 0.01, 0.1, 1]:
    #         model = AutoEncoder(num_qs, k)
    #         _lambda = 0.001
    #         print(f'======== k: {k}, lr: {lr}, lambda: {_lambda} ========')
    #         train(model, lr, _lambda, train_matrix, zero_train_matrix,
    #             valid_data, num_epoch)
    #
    #         print(f'Test Accuracy: {evaluate(model, zero_train_matrix, test_data)[0]}')
    #
    # print(f'======== k: {k}, lr: {lr}, lambda: {_lambda} ========')
    
    # Final run with optimized hyperparameters, and regularization
    k_star = 50
    lr_star = 0.01
    
    print(f'OPTIMAL --> k: {k_star}, lr: {lr_star}')
    for lbda in [0.001, 0.01, 0.1, 1]:
        print(f'======== lambda: {lbda} ========')
        model_star = AutoEncoder(num_qs, k_star)
        train(model_star, lr_star, lbda, train_matrix, zero_train_matrix,
            valid_data, num_epoch)
        print(f'Test Accuracy: {evaluate(model_star, zero_train_matrix, test_data)[0]}')

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
