import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse

from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc

import keras
from keras import optimizers
from keras.utils import plot_model
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import BatchNormalization

def read_data(args):	
	x_train = pd.read_csv(args.x_train_file).iloc[:,1:].values
	y_train = pd.read_csv(args.y_train_file).iloc[:,-1:].values
	x_test = pd.read_csv(args.x_test_file).iloc[:,1:].values
	y_test = pd.read_csv(args.y_test_file).iloc[:,-1:].values
	return x_train, x_test, y_train, y_test

def plot_training_curve(history, args):
	key = args.y_train_file.replace('data/','').replace('.csv','')
	figure_name = 'results/training_curve/acc_' + key + '_' + str(args.num_hidden_layers) + '_' + str(args.epochs) + '.eps'
	plt.figure()
	plt.plot(history.history['accuracy'])
	plt.plot(history.history['val_accuracy'])
	plt.title('Model accuracy')
	plt.ylabel('Accuracy')
	plt.xlabel('Epoch')
	plt.legend(['Train', 'Validation'], loc='upper left')
	plt.grid()
	plt.savefig(figure_name)

	figure_name = 'results/training_curve/loss_' + key + '_' + str(args.num_hidden_layers) + '_' + str(args.epochs) + '.eps'
	plt.figure()
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('Model loss')
	plt.ylabel('Loss')
	plt.xlabel('Epoch')
	plt.legend(['Train', 'Validation'], loc='upper left')
	plt.grid()
	plt.savefig(figure_name)

def plot_model_diagram(model, args):
	key = args.y_train_file.replace('data/','').replace('.csv','')
	figure_name = 'model_diagram/' + key + '_' + str(args.num_hidden_layers) + '_' + str(args.epochs) + '.png' 
	plot_model(model, show_shapes = True, to_file = figure_name )

def get_data(args):
	x, x_test, y, y_test = read_data(args)

	# normalization of the data
	sc = StandardScaler()
	x = sc.fit_transform(x)
	x_test = sc.fit_transform(x_test)

	# create one hot encoded data
	ohe = OneHotEncoder()
	y = ohe.fit_transform(y).toarray()
	y_test = ohe.fit_transform(y_test).toarray()

	# split the data for training and validation
	x_train, x_val, y_train, y_val = train_test_split(x,y,stratify = y,test_size = 0.2)
	return x_train, x_val, x_test, y_train, y_val, y_test

def create_model(input_shape, output_shape, args):
	model = Sequential()
	model.add(Dense(128, input_dim = input_shape, activation = args.activation))
	for i in range(args.num_hidden_layers):
		model.add(Dense(128, activation = args.activation))
		model.add(BatchNormalization())
		model.add(Dropout(args.dropout))
	model.add(Dense(output_shape, activation = "softmax"))
	return model

def evaluate(x_test, y_test, model, args):
	y_pred = model.predict(x_test)
	y_score = np.argmax(y_pred,1)
	y_true = np.argmax(y_test,1)
	acc = accuracy_score(y_true, y_score)
	
	key = args.y_train_file.replace('data/','').replace('.csv','')
	file_name = 'results/' + key + '_' + str(args.num_hidden_layers) + '_' + str(args.epochs) + '.txt'
	f = open(file_name, "w")
	f.write(str(acc) + '\n\n')
	f.write(str(confusion_matrix(y_true,y_score)) + '\n\n')
	f.write(str(classification_report(y_true,y_score)))
	f.close()
	
	fpr = dict()
	tpr = dict()
	roc_auc = dict()
	for i in range(4):
		fpr[i], tpr[i], _ = roc_curve(y_test[:,i], y_pred[:,i])
		roc_auc[i] = auc(fpr[i], tpr[i])
	plt.figure()
	figure_name = 'results/roc_curve/' + key + '_' + str(args.num_hidden_layers) + '_' + str(args.epochs) + '.eps'
	plt.plot(fpr[0], tpr[0], label = 'ROC Curve for class 0 (area = %0.2f)' % roc_auc[0])
	plt.plot(fpr[1], tpr[1], label = 'ROC Curve for class 1 (area = %0.2f)' % roc_auc[1])
	plt.plot(fpr[2], tpr[2], label = 'ROC Curve for class 2 (area = %0.2f)' % roc_auc[2])
	plt.plot(fpr[3], tpr[3], label = 'ROC Curve for class 3 (area = %0.2f)' % roc_auc[3])
	plt.plot([0, 1], [0, 1], color='navy',linestyle='--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('ROC Curve')
	plt.legend(loc="lower right")
	plt.grid()
	plt.savefig(figure_name)
	print acc
	print classification_report(y_true,y_score)

def execute(args):
	x_train, x_val, x_test, y_train, y_val, y_test = get_data(args)
	model = create_model(x_train.shape[1], y_train.shape[1], args)
	if args.optimizer == 'adam':
		optimizer_type = optimizers.Adam(learning_rate = args.lr)
	elif args.optimizer == 'rmsprop':
		optimizer_type = optimizers.RMSprop(learning_rate = args.lr)
	elif args.optimizer == 'sgd':
		optimizer_type = optimizers.SGD(learning_rate = args.lr, nesterov=True)
	model.compile(loss = args.loss, optimizer = optimizer_type, metrics=['accuracy'])
	history = model.fit(x_train, y_train, validation_data = (x_val, y_val), epochs = args.epochs, batch_size = args.batch_size)
	evaluate(x_test,y_test, model, args)
	plot_training_curve(history,args)
	plot_model_diagram(model, args)
	#print evaluate(x_test, y_test, model)
	#print confusion_matrix()

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--x_train_file', action = "store", dest = "x_train_file", default = "data/x_train.csv", type = str)
	parser.add_argument('--x_test_file', action = "store", dest = "x_test_file", default = "data/x_test.csv", type = str)
	parser.add_argument('--y_train_file', action = "store", dest = "y_train_file", default = "data/y_180_train.csv", type = str)
	parser.add_argument('--y_test_file', action = "store", dest = "y_test_file", default = "data/y_180_test.csv", type = str)

	parser.add_argument('--activation', action = "store", dest = "activation", default = "relu", type = str)
	parser.add_argument('--loss',action = "store", dest = "loss", default = "categorical_crossentropy", type = str)
	parser.add_argument('--batch_size', action = "store", dest = "batch_size", default = 64, type = int)
	parser.add_argument('--epochs', action = "store", dest = "epochs", default = 20, type = int)
	parser.add_argument('--lr', action="store", dest="lr", default=0.001, type = float)
	parser.add_argument('--num_hidden_layers', action = "store", dest = "num_hidden_layers", default = 3, type = int)
	parser.add_argument('--optimizer', action = "store", dest = "optimizer", default = 'adam', type = str)
	parser.add_argument('--dropout', action = "store", dest = "dropout", default = 0.0, type = float)

	args = parser.parse_args()
	execute(args)