"""
This script is used to help with analyzing changes
to our neural network for part B

Things like plotting data or findning maximums, etc.
"""
import os
import re
import matplotlib.pyplot as plt
from typing import List

def read_data(datafile: str) -> dict:
	epochs = []
	training_costs = []
	valid_accs = []
	test_accs = []
	valid_costs = []
	with open(datafile, 'r') as f:
		for line in f:
			# line format:
			# Epoch:{}:TrainingCost:{}:ValidAcc:{}:TestAcc:{}}
			line.strip()
			if len(line) == 0:
				continue
			tokens = line.split(':')
			epochs.append(int(tokens[1]))
			training_costs.append(float(tokens[3]))
			valid_accs.append(float(tokens[5]))
			test_accs.append(float(tokens[7]))
			valid_costs.append(float(tokens[9]))
	return {
		'epochs': epochs,
		'training_costs': training_costs,
		'valid_accs': valid_accs,
		'test_accs': test_accs,
		'valid_costs': valid_costs
	}

def plot_data(data: dict, keys: List[str], y_label: str, title: str, color_start_index=0) -> None:
	colors = ['red', 'green', 'blue', 'orange', 'purple', 'black', 'gray']
	for i, key in enumerate(keys):
		plt.plot(data['epochs'], data[key], color=colors[(i+color_start_index)%len(colors)], label=key)
	plt.xlabel('Epochs')
	plt.ylabel(y_label)
	plt.title(title)
	plt.legend()
	plt.show()

def main():
	# Edit parameters below
	# Epsilon symbol for easy Ctrl-C: ε
	file_name = 'early_stopping_1000.txt'
	graph_name = 'Early Stopping, ε=1000.0'

	## Plotting code below
	data = read_data(os.path.join(os.curdir, 'data', file_name))
	plot_data(
		data, 
		['valid_accs', 'test_accs'], 
		'Accuracy', 
		f"Valid vs Test Accuracy ({graph_name})",
	)
	plot_data(
		data, 
		['valid_costs', 'training_costs'], 
		'Cost', 
		f"Valid vs Training Cost ({graph_name})",
		color_start_index=2
	)


if __name__ == "__main__":
	main()
