#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from sklearn import preprocessing
from sklearn.utils import shuffle

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
	dtype = torch.cuda.FloatTensor
else:
	dtype = torch.FloatTensor

    
def makegrid(numberoflayers, hiddenlayersizes, batchsizes, dropouts, learningrates, lookbacks):
	gridpoints = []
	for hiddenlayer in hiddenlayersizes:
		for batch in batchsizes:
			for learningrate in learningrates:
				for lookback in lookbacks:
					gridpoint = [int(1), int(hiddenlayer), int(batch), 0, learningrate, int(lookback)]
					gridpoints.append(gridpoint)
	for number in range(2, len(numberoflayers)+1):
		for hiddenlayer in hiddenlayersizes:
			for batch in batchsizes:
				for learningrate in learningrates:
					for lookback in lookbacks:
						for dropout in dropouts:
							gridpoint = [int(number), int(hiddenlayer), int(batch), dropout, learningrate, int(lookback)]
							gridpoints.append(gridpoint)
	return gridpoints


text_file_19982012 = '19982012.pkl'
text_file_20042007 = '20042007.pkl'
text_file_20122014 = '20122014.pkl'

#Defines the amount of hours later we want to forecast the Dst
forecasthour = 5

#maximum total amount of epochs
#do not take this tooo low because when error on CV will not increase (when locl minimum, when learning rate too small)
num_epochs = 400

input_size = 5
output_size = 1

#hyperparameters grid
num_layers_grid = [1,2,3] #Number of HIDDEN LSTM layers
hidden_layer_size_grid = [16,32,64,128,256] #Number of features in lstm hidden state
learning_rate_grid = [0.001, 0.0005, 0.0001] #learning rate
lookback_grid = [15, 30, 45, 60] #lookback = sequence length
batch_size_grid = [128, 256, 512, 1024] #size of amount of sequences in each batch
dropout_grid = [0] #dropout, only in networks with more layers

hyperparametergrid = makegrid(num_layers_grid, hidden_layer_size_grid, batch_size_grid, dropout_grid,
                              learning_rate_grid, lookback_grid)


print('amount of grid points', len(hyperparametergrid))


# load in the scaler data 1998-2012

datapanda = pd.read_pickle(text_file_19982012)

#Quentifify total number of parameters - timestamps
NrParam = datapanda.values[0].shape[0] - 3
#get all the input parameters including Dst (without timestamps)
inputdata = datapanda.loc[:,3:(2+NrParam)].values
#get the Dst output
outputdata = datapanda.loc[:,6].values
outputdata = outputdata.reshape(-1, 1)

#use the preprocessing tool of sklearn to rescale data between -1 and 1
inputscaler = preprocessing.StandardScaler()
inputscaler.fit(inputdata)
#save scaler to later inverse the scaled result
scaler = preprocessing.StandardScaler()
scaler.fit(outputdata)

# load in the training data 2004-2007

datapanda = pd.read_pickle(text_file_20042007)

#Quentifify total number of parameters - timestamps
NrParam = datapanda.values[0].shape[0] - 3
#get all the input parameters including Dst (without timestamps)
traininginput = datapanda.loc[:,3:(2+NrParam)].values
#get the Dst output
trainingoutput = datapanda.loc[:,6].values
trainingoutput = trainingoutput.reshape(-1, 1)

# load in the cv data 2012-2014

datapanda = pd.read_pickle(text_file_20122014)

#Quentifify total number of parameters - timestamps
NrParam = datapanda.values[0].shape[0] - 3
#get all the input parameters including Dst (without timestamps)
CVinput = datapanda.loc[:,3:(2+NrParam)].values
#get the Dst output
CVoutput = datapanda.loc[:,6].values
CVoutput = CVoutput.reshape(-1, 1)

traininginput = inputscaler.transform(traininginput)
CVinput = inputscaler.transform(CVinput)
trainingoutput = scaler.transform(trainingoutput)
CVoutput = scaler.transform(CVoutput)


def create_dataset(datasetinput, datasetoutput, look_back, forecast):
	dataX, dataY = [],[]
	for i in range(len(datasetinput) - int(look_back) - int(forecast) - 1):
		x = datasetinput[i:(i+int(look_back))]
		dataX.append(x)
		y = datasetoutput[(i+int(forecast)):(i+int(look_back)+int(forecast))]
		dataY.append(y)
	return np.array(dataX), np.array(dataY), len(dataY)

def prepare_data(inputdata, outputdata, lookback, forecast, batchsize):
	lookback = int(lookback)
	forecast = int(forecast)
	batchsize = int(batchsize)

	numberofdata = len(inputdata)
	number_of_batches = (numberofdata // batchsize)

	#create a list where we will put all the batches in
	Xlist = []
	Ylist = []

	dataX, dataY, numberdummy = create_dataset(inputdata, outputdata, lookback, forecast)

	#random shuffle the samples 
	indices = np.arange(dataX.shape[0])
	np.random.shuffle(indices)
	dataX = dataX[indices]
	dataY = dataY[indices]

	for b in range(0,number_of_batches-1):
		Xbatch2 = dataX[(b*batchsize):((b+1)*batchsize)]
		Ybatch2 = dataY[(b*batchsize):((b+1)*batchsize)]
		Xtensor2 = torch.Tensor(Xbatch2).type(dtype)
		#transpose because we want shape (seq length, batchsize, number of features)
		Xtensor2 = Xtensor2.transpose(1,0)
		Ytensor2 = torch.Tensor(Ybatch2).type(dtype)
		Ytensor2 = Ytensor2.transpose(1,0)
		#add tensor batches to lists of all batches
		Xlist.append(Xtensor2)
		Ylist.append(Ytensor2)   

	return Xlist, Ylist



# Here we define our model as a class
class lstm(nn.Module):
	def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout):
		super(lstm, self).__init__()
		self.input_dim = input_dim
		self.hidden_dim = hidden_dim
		self.num_layers = num_layers
		self.dropout = dropout

		# Define the LSTM layer(s)
		self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers, dropout = self.dropout)

		# Define the output layer
		self.linear = nn.Linear(self.hidden_dim, output_dim)

	def forward(self, input):
		batch_size = input.size(1)
		hidden = torch.zeros(self.num_layers, batch_size, self.hidden_dim).type(dtype), torch.zeros(self.num_layers, batch_size, self.hidden_dim).type(dtype)
		# Forward pass through LSTM layer
		# shape of lstm_out: [input sequence length, batch_size, hidden_dim (features)]
		# shape of self.hidden: (a, b), where a and b both have shape (num_layers, batch_size, hidden_dim).
		lstm_out, hidden = self.lstm(input.view(len(input), batch_size, -1))
		y_pred = self.linear(lstm_out)
		return y_pred

'''
def MSE_last(pred, target):
	error = 0
	for i in range(0, pred.shape[1]):
		errorpart = (target[-1, i, 0] - pred[-1, i, 0])**2
		error = error + errorpart
	error = error/pred.shape[1]
'''
#We use MSE loss
loss_fn = nn.MSELoss()

currentsmallest = 0.6
point = 1

model_1_back = 0
model_2_back = 0

for parameters in hyperparametergrid:
	print('Next grid point', point)
	num_layers, hidden_layer_size, batch_size, dropout, learning_rate, lookbackhour = parameters
	#prepare the data with the function above
	trainX, trainY = prepare_data(traininginput, trainingoutput, lookbackhour, forecasthour, batch_size)
	cvX, cvY = prepare_data(CVinput, CVoutput, lookbackhour, forecasthour, batch_size)
	#define new model
	model = lstm(input_size, hidden_layer_size, output_dim=output_size, num_layers=num_layers, dropout=dropout)
	optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
	if torch.cuda.is_available():
		model = model.cuda()
	histcv = np.zeros(num_epochs)
	#model.train()
	for e in range(num_epochs):
		hist3 = np.zeros(len(cvX))
		for l in range(0, len(cvX)):
			predcv = model(cvX[l])
			targ = cvY[l]
			losscv = loss_fn(predcv[-1], targ[-1])
			hist3[l] = losscv.item()
		#print(np.average(hist3))
		histcv[e] = np.average(hist3)

		for t in range(0, len(trainX)):
			#Do I have to uncomment? zero grad?   
			#model.zero_grad()
			prediction = model(trainX[t])
			loss = loss_fn(prediction, trainY[t])
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			#print("Epoch: ", e, ", Batch: ", t, " Loss: ", loss.item())
	

		#if we are in the first 3 loops skip the next step
		if e<2:
			continue
		#stop when average CV loss has been increasing for two epochs
		if histcv[e]>histcv[e-1] and histcv[e-1]>histcv[e-2]:
			break
	histcv = histcv[0:e]

	error = histcv[e-2]
	print('Error', error)
	print(parameters)
	print('Epochs:', e)

	if error < currentsmallest:
		currentsmallest = error
		parameters_optim = parameters
		optim_epochs = e-2
	point = point + 1


print(parameters_optim)    
print('Optimal parameters dataset 2 ', forecasthour, ' in advance ',  parameters_optim)
print('Optimal amount of epochs', optim_epochs)









