from warnings import filters
import numpy as np
# univariate multi-step encoder-decoder convlstm
from math import sqrt
from numpy import split
from numpy import array
from pandas import read_csv
from sklearn.metrics import log_loss, mean_squared_error
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import LSTM
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.layers import ConvLSTM2D
from keras.layers import BatchNormalization
from tensorflow import keras
from numpy.fft import fft, ifft
import math
import datetime as dt
from sklearn.model_selection import PredefinedSplit
import statsmodels
from statsmodels.tsa.stattools import acf

# My packages
from packages.train_data_no_minmax import train_data

'''
The class build model has the goal to build the model to be used for prediction
dataset : DataFrame
n_steps : How many steps for each fit. Example: 96 hourly load in one step = 96 hours array, 96 hours in 2 steps = 2x48 arrays
n_length : How many hourly loads in each step = n_input * n_steps
n_input : How many hourly loads in each fit

'''

class build_model:
    def __init__(self, dataset, n_steps, n_length, n_input,n_features, n_out, epochs, batch_size, filters, activation, loss, optimizer,karnel,dense_1,dense_2):
            self.dataset = dataset
            self.n_steps = n_steps
            self.n_length = n_length
            self.n_input = n_input
            self.n_features = n_features
            self.n_out = n_out
            self.epochs = epochs
            self.batch_size = batch_size
            self.filters = filters
            self.activation = activation
            self.loss = loss
            self.optimizer = optimizer
            self.karnel = karnel
            self.dense_1 = dense_1
            self.dense_2 = dense_2

    def Conv_LSTM_model(self):
        # Transform train into train_x and train_y
        train = train_data(self.dataset, self.n_input, self.n_features, self.n_out)
        train_x, train_y = train.transform()
        # reshape into subsequences [samples, time steps, rows, cols, channels]
        train_x = train_x.reshape((train_x.shape[0], self.n_steps, self.n_length, self.n_features, 1))            
        # reshape output into [samples, timesteps, features]
        train_y = train_y.reshape((train_y.shape[0], train_y.shape[1], 1))       
        # define model
        model = Sequential()
        model.add(ConvLSTM2D(filters=self.filters, return_sequences=True, kernel_size=self.karnel, activation=self.activation, input_shape=(self.n_steps, self.n_length, self.n_features,1)))
        model.add(Flatten())
        model.add(Dense(self.dense_1, activation= self.activation))
        model.add(Dense(self.dense_2, activation= self.activation))
        model.compile(loss=self.loss, optimizer=self.optimizer) 
        # fit network
        model.fit(train_x, train_y, epochs=self.epochs, batch_size=self.batch_size, verbose = 0)
        return model

class autocorrelation:
    def __init__(self, train_x):
        self.train_x = train_x       
    def get_highest_power_spectrum(self, n_input):
        #get hourly load sequence and invert sequence since the last position is the first day in the time series
        sequence = self.train_x[:,0][::-1]
        # get power spectrum: y(t) -> FFT -> 1/N(log|.|^2) -> IFFT -> cy(k)
        # FFT
        df_fft = fft(sequence)      
        # 1/N(log|.|^2)
        df_log = (1/(len(sequence)))*(np.log10(df_fft**2))
        # IFFT -> cy(k)
        power_spectrum = abs(ifft(df_log))
        # Standarize acf and invert sequence to come back to (y-n to y)
        power_spectrum_std = ((power_spectrum - np.mean(power_spectrum))/np.std(power_spectrum) + 0.5)[1:][::-1]
        # get indexes from highest power_spectrum
        best_index_power_spectrum = np.sort(np.argpartition(power_spectrum_std, -n_input)[-n_input:])
        # min power spectrum value without first autocorrolation
        min_power_spectrum = power_spectrum_std[best_index_power_spectrum].min()
        return min_power_spectrum, best_index_power_spectrum

    def get_highest_acf(self, n_input):
        sequence = self.train_x[:,0]
        # absolut number for acf to get highest autocorrelation and invert sequence to have the correct order (y to y-n)
        acf = abs(statsmodels.tsa.stattools.acf(sequence[::-1], nlags= len(sequence)))
        # Standarize acf and invert sequence to come back to (y-n to y)
        acf_std = ((acf - np.mean(acf))/np.std(acf) + 0.5)[1:][::-1]
        # get indexes from highest power_spectrum
        best_index_acf = np.sort(np.argpartition(acf_std, -n_input)[-n_input:])
        # min power spectrum value without first autocorrolation
        min_acf = acf_std[best_index_acf].min()
        return min_acf, best_index_acf

class evaluate_model:
    def __init__(self, predicted_list, observed_list):
        self.predicted_list = predicted_list
        self.observed_list = observed_list
    
    def evaluate_model(self):
        scores = list()
        score_overall = list()
	    # calculate an RMSE score for each day
        for i in range(len(self.observed_list)):
	    	# calculate mse
            mse = (self.observed_list[i]-self.predicted_list[i])**2
            # store
            scores.append(sqrt(mse))
	    # calculate overall RMSE
            score_overall.append(mse)
        s = 0
        s = np.array(score_overall,dtype=object).sum()
        score = sqrt(s / (len(self.observed_list)))/np.mean(self.observed_list)
        return score, scores



