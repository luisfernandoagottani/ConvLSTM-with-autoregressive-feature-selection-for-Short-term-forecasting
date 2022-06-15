from enum import auto
import pandas as pd
import numpy as np
from sklearn.model_selection import learning_curve
from packages.thesis_model_nominmax import build_model
from packages.thesis_model_nominmax import autocorrelation
from sklearn import preprocessing
from tensorflow import keras
import time

class thesis_sequence_algorithm:
    def __init__(self, dataset, autocorrelation_type, warm_up_time, theta_threshold, leg_days_time, n_steps, n_length, n_input,n_features, n_out, epochs, batch_size, filters, activation, loss, optimizer, karnel, epochs_retrain,dense_1,dense_2):
        self.dataset = dataset
        self.autocorrelation_type = autocorrelation_type
        self.warm_up_time = warm_up_time 
        self.theta_threshold = theta_threshold
        self.leg_days_time = leg_days_time
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
        self.epochs_retrain = epochs_retrain
        self.dense_1 = dense_1
        self.dense_2 = dense_2

    def master_algorithm(self):
        # Create lists for evaluation
        start_time = time.time()
        predict_list = []
        observed_list = []
        # Prepare warm up period
        dataset_warm_up = self.dataset[:self.warm_up_time]
        # set model parameters
        model_parameters = build_model(dataset_warm_up, self.n_steps, self.n_length, self.n_input, self.n_features, self.n_out, self.epochs, self.batch_size, self.filters, self.activation, self.loss, self.optimizer,self.karnel, self.dense_1, self.dense_2)
        # build model ConvLSTM with warm up data
        model = model_parameters.Conv_LSTM_model()
        # Transform dataset into array
        dataset_array = np.array(self.dataset)
        print((time.time() - start_time))
        # Period of training and testing simulating normal days after warm up
        start_time = time.time()
        for i in range((self.warm_up_time),len(self.dataset)):
            # Split data into train_x and train_y for prediction and traning
            train_x = dataset_array[(i-self.leg_days_time):i]
            train_y = dataset_array[i:(i+self.n_out),0]
            observed_list.append(dataset_array[i:(i+self.n_out),0][0])
            # Select optimal lagged load by the power spectrum
            if self.autocorrelation_type == "power_spectrum":
                dataset_autocorrelation = autocorrelation(train_x)
                max_power_spectrum, max_index_power_spectrum = dataset_autocorrelation.get_highest_power_spectrum(self.n_input)
                # Correlation of the lag > Theta?
                if max_power_spectrum > self.theta_threshold:
                    # Reshape the input vector with the best lag for ConvLSTM model
                    input_x = np.concatenate([train_x[max_index_power_spectrum][0][:4],train_x[-self.n_out:][0][4:]])
                    input_x = input_x.reshape((1, self.n_steps, self.n_length, self.n_features,1))
                    # Implement single step ahead forecast by the ConvLSTM model
                    y_hat = model.predict(input_x, verbose=0)
                    # Append predicted load for model evaluation
                    predict_list.append(y_hat[0][0])
                else:
                    # Implement Single Step ahead Forecast by Persistence model
                    # Append predicted value for evaluation list
                    predict_list.append(dataset_array[(i-self.n_out):i,0])
            # elif self.autocorrelation_type == "acf":
            #     dataset_autocorrelation = autocorrelation(train_x)
            #     max_acf, max_index_acf = dataset_autocorrelation.get_highest_acf(self.n_input)
            #     # Correlation of the lag > Theta?
            #     if max_acf > self.theta_threshold:
            #         input_x = train_x[max_index_acf]
            #         input_x = input_x.reshape((1, self.n_steps, self.n_length, self.n_features,1))
            #         # Implement single step ahead forecast by the ConvLSTM model
            #         y_hat = model.predict(input_x, verbose=0)
            #         # Append predicted load for model evaluation
            #         predict_list.append(y_hat[0][0])
            #     else:
            #         # Implement Single Step ahead Forecast by Persistence model
            #         # Append predicted value for evaluation list
            #         predict_list.append(dataset_array[(i-self.n_out):i,0])
            #Retrain the ConvLSTM model by the observed hourly load of previus predicted time step
            #Retrain model by last hourly leg
            train_x = train_x[-self.n_input:].reshape((1, self.n_steps, self.n_length, self.n_features,1))
            model.fit(train_x, train_y, epochs=self.epochs_retrain, batch_size=1, verbose  = 0)
        print((time.time() - start_time))
        return predict_list, observed_list

    def master_algorithm_convlstm(self):
        # Create lists for evaluation
        start_time = time.time()
        predict_list = []
        observed_list = []
        # Prepare warm up period
        dataset_warm_up = self.dataset[:self.warm_up_time]
        # set model parameters
        model_parameters = build_model(dataset_warm_up, self.n_steps, self.n_length, self.n_input, self.n_features, self.n_out, self.epochs, self.batch_size, self.filters, self.activation, self.loss, self.optimizer,self.karnel, self.dense_1, self.dense_2)
        # build model ConvLSTM with warm up data
        model = model_parameters.Conv_LSTM_model()
        # Transform dataset into array
        dataset_array = np.array(self.dataset)
        print((time.time() - start_time))
        # Period of training and testing simulating normal days after warm up
        start_time = time.time()
        for i in range((self.warm_up_time),len(self.dataset)):
            # Get train_y
            train_y = dataset_array[i:(i+self.n_out),0]
            # Append observed load
            observed_list.append(dataset_array[i:(i+self.n_out),0][0])
            # Get input for prediction
            input_x = dataset_array[-self.n_input:]
            # Reshape
            input_x = input_x.reshape((1, self.n_steps, self.n_length, self.n_features,1))
            # Predict
            y_hat = model.predict(input_x, verbose=0)
            # Append predicted load for model evaluation
            predict_list.append(y_hat[0][0])
            #Retrain model by last hourly leg
            train_x = input_x.reshape((1, self.n_steps, self.n_length, self.n_features,1))
            model.fit(train_x, train_y, epochs=self.epochs_retrain, batch_size=1, verbose  = 0)
        print((time.time() - start_time))
        return predict_list, observed_list