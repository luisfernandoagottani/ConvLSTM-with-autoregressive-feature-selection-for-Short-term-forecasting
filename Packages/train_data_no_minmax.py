import numpy as np
from sklearn import preprocessing

'''
The train_data class has the function transform where it takes the dataset
and transform into array with the correct shape defined by:

n_input: How many hourly steps are going to be used for fit
n_features: How many features are going to be used for fit
n_out: How many hourly steps are going to be predicted
'''

class train_data:
	def __init__(self, dataset, n_input,n_features, n_out):
		self.dataset = np.array(dataset)
		self.n_input = n_input
		self.n_features = n_features
		self.n_out = n_out

	def transform(self):
		train_x_scaled = self.dataset
		train_y_scaled = self.dataset[:,0]
		# Create list
		train_x, train_y = list(), list()
		input_start = 0
		# Fit data to scaler
		# step over the entire history one time step at a time
		for i in range(len(self.dataset)):
			# define the end of the input sequence
			input_end = input_start + self.n_input
			out_end = input_end + self.n_out
			# ensure we have enough data for this instance
			if out_end <= len(self.dataset):
				#train_x.append(train_x_scaler.transform(x_input))
				train_x.append(train_x_scaled[input_start:input_end, :self.n_features])
				train_y.append(train_y_scaled[input_end:out_end])
			# move along one time step
			input_start += 1
		return np.array(train_x), np.array(train_y)
