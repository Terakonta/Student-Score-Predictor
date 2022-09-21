"""
This script is used to help with developing the neural network 
for CSC311 project
It is just a collection of functions to do things like:
 - finding best k*
 - plotting
 - etc.

Written by Stew Esho
"""
import os
import re
import matplotlib.pyplot as plt

def avg(l: list) -> float:
	return sum(l)/len(l)

def get_k_from_label(label: str) -> int:
	return int(label.split('k: ')[1].split(', ')[0])

def get_lr_from_label(label: str) -> float:
	return float(label.split('lr: ')[1].split(', ')[0])

def find_k_star(file_name: list) -> int:
	print("Processing for k* . . . ")
	k_to_lr = dict()
	curr_k = 0
	curr_lr = 0.0

	curr_max_precision = 0.0
	curr_epoch = ""
	curr_max_epoch = ""
	
	k_star = 0
	lr_star = 0.0
	epoch_star = ""
	max_precision = 0.0

	with open(file_name, 'r') as f:
		for line in f:
			line.strip()
			if line[0] == '=':
				# ======== k: <k>, lr: <learning_rate>, lambda: <lambda> ========
				print(f'k: {curr_k}, lr: {curr_lr}, epochs: {curr_max_epoch}|| acc: {curr_max_precision}')
				if curr_max_precision > max_precision:
					max_precision = curr_max_precision
					k_star = curr_k
					lr_star = curr_lr
					epoch_star = curr_max_epoch

				if curr_k not in k_to_lr:
					k_to_lr[curr_k] = dict()
				k_to_lr[curr_k][curr_lr] = curr_max_precision
				curr_k = get_k_from_label(line)
				curr_lr = get_lr_from_label(line)
				curr_max_epoch = ''
				curr_max_precision = 0.0
			elif line[0] == 'T': 
				# Test Accuracy: <test_acc> 
				pass # we don't need the test accuracy here, so ignore it
			else:
				# Epoch: <epoch> 	Training Cost: <cost>	 Valid Acc: <acc>
				tokens = line.split(': ')
				try:
					curr_epoch = tokens[1].split(' \t')[0]
					curr_precision = float(tokens[-1])
				except:
					curr_epoch = '-'
					curr_precision = 0.0
				finally:
					if curr_precision > curr_max_precision:
						curr_max_precision = curr_precision
						curr_max_epoch = curr_epoch
	
	print(f'kstar: {k_star} | lr_star: {lr_star} | epoch_star: {epoch_star} || precision: {max_precision}')
	return

def plot_precision_by_epochs(datafile: str) -> None:
	# first get data
	epochs = []
	training_costs = []
	valid_accs = []
	with open(datafile, 'r') as f:
		for line in f:
			line.strip()
			if len(line) == 0:
				continue
			if line[0] == '#':
				continue
			tokens = re.split('[^\w.]+', line) 	# format: label, epoch, label, t_cost, label, 
			epochs.append(int(tokens[1]))
			training_costs.append(float(tokens[3]))
			valid_accs.append(float(tokens[5]))

	print(max(valid_accs))
	print(valid_accs.index(max(valid_accs)))		
	
	# plt.plot(epochs, valid_accs, color='red')
	plt.plot(epochs, training_costs, color='blue')
	plt.title('NN Accuracy over Time (k=10, alpha=0.05)')
	plt.xlabel('Epochs')
	plt.ylabel('Training Cost')
	# plt.ylabel('Validation Accuracy')
	plt.show()

def main():
	find_k_star(os.path.join(os.curdir, 'notes', f'new_data.txt'))
	print('----------------------------')
	plot_precision_by_epochs(os.path.join(os.curdir, 'notes', 'plot_data.txt'))

if __name__ == "__main__":
	main()
