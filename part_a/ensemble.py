# TODO: complete this file.
from utils import *
import sklearn
import scipy
import numpy as np
import pandas as pd
import item_response as irt
import knn
import neural_network as nn
import torch

def bootstrap(train_data,i):
	""" Bootstrap the dataset using sklearn.utils.resample

	:param train_data: dict
	:param i: int
	:return: dict
	"""

	dframe = pd.DataFrame.from_dict(train_data)
	dframe = sklearn.utils.resample(dframe,random_state=i)
	return pd.DataFrame.to_dict(dframe, orient='list')

def bag(bagged_data,val_data):
	irt_hyperparameters = {
        "lr": 0.01,
        "iterations": 50
    }

	nn_hyperparameters = {"lr": 0.05,"num_epoch": 13,"k_star": 10,"num_qs": 1774}
	bagged_matrix = build_sparse_mat(bagged_data)

	knn_matrix = knn.knn_impute_by_item(bagged_matrix.toarray(), val_data, 16)[1]
	knn_pred = sparse_matrix_predictions(val_data,knn_matrix.T)

	bagged_theta = np.zeros(shape=(542,1))
	bagged_beta = np.zeros(shape=(1774,1))
	bagged_theta, bagged_beta = irt.irt(bagged_matrix,val_data, irt_hyperparameters["lr"],irt_hyperparameters["iterations"])[:2]
	irt_pred = sparse_matrix_predictions(val_data, irt.sigmoid(bagged_theta - bagged_beta.T))


	zero_train_matrix, train_matrix, valid_data, test_data = nn.load_data()
	bagged_data = torch.FloatTensor(bagged_matrix.toarray())
	model = nn.AutoEncoder(nn_hyperparameters["num_qs"], nn_hyperparameters["k_star"])
	nn.train(model, nn_hyperparameters["lr"], 0.001, bagged_data, zero_train_matrix, valid_data, nn_hyperparameters["num_epoch"])
	nn_pred = nn.evaluate(model, bagged_data, valid_data)[1]
	
	true_predictions = np.add(knn_pred,irt_pred)
	true_predictions = np.add(true_predictions,nn_pred)

	true_predictions = np.where(true_predictions > 1.0, 1.0, 0.0)

	return true_predictions


def main():
	train_data = load_train_csv("../data")
	val_data = load_valid_csv("../data")
	test_data = load_public_test_csv("../data")
	
	for i in [16]:
		bagged_data = bootstrap(train_data, i)
		final_pred = bag(bagged_data,val_data)
		acc = evaluate(val_data, final_pred)
		print(acc)



if __name__ == "__main__":
    main()
