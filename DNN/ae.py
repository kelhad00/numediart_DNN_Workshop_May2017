import tensorflow as tf
import numpy as np
import sys
import os
import librosa
from shutil import copyfile
from random import shuffle
from os import listdir
from os.path import join, isfile
from copy import deepcopy

''' This code reads data from readable files, concatenates them in one matrix and uses it to train an MLP or AE'''

path = sys.argv[1]

file_extension = 'mcep' #feature files extension
others_ext = 'pitch' #the remaining extension
neutral_dir = '' #source files directory
smile_dir = '' #traget files directory

prev = 1 #previous frames to add
fol = 1 #following frames to add

learning_rate = 0.01
epochs = 10000
batch_size = 200
display_step = 1
number_of_tests = 5

def read_data_in_text_as_list(path, sep = '\t'):
	''' Read features from text files as lists of float '''
	f = open(path, 'r')
	l = f.readlines()
	l = [x.strip('\n').split(sep) for x in l] #remove \n and split wrt space
	l = [list(map(float, x)) for x in l] # convert from list of lists to array
	return l


def concat_to_mat(path, lst):
	''' Concatenate features from files in path+lst into np.array '''
	concat_mat = []
	for l in lst:
		mat = read_data_in_text_as_list(path + l)
		concat_mat = concat_mat + mat
	return np.array(concat_mat)

def concat_to_mat_with_context(path, lst, prev, fol):
	''' Concatenate features from files in path+lst into np.array
	after adding 'prev' previous frames and 'fol' following frames '''
	for l in lst:
		mat = read_data_in_text_as_list(path + l)
		mat = np.array(mat)
		mat = add_context(mat, prev, fol)

		if "concat_mat" not in locals():
			concat_mat = deepcopy(mat)
		else:
			concat_mat = np.concatenate((concat_mat, mat), axis = 0)
	return concat_mat


def add_context(mat, prev, fol):
	''' mat is the initial numpy array
	prev is an integer of the number of previous frames to consider
	fol is an integer of the number of following frames to consider'''
	
	right = np.zeros( [mat.shape[0], (prev * mat.shape[1])] )

	for i in range(right.shape[0]):
		if i < prev:
			z = np.zeros([1, (prev - i)*mat.shape[1]])
			if z.shape[1] == right.shape[1]:
				right[i,:] = z
			else:
				right[i,:] = np.concatenate((z, np.reshape(mat[:i,:], (1, i*mat.shape[1]))), axis = 1)
		else:
			right[i,:] = np.reshape(mat[(i-prev) : i,:], (1, mat.shape[1]*prev))


	left = np.zeros( [mat.shape[0], (fol * mat.shape[1])] )

	for i in range(left.shape[0]):
		if i > (mat.shape[0]-1-fol):
			z = np.zeros([1, (i - (mat.shape[0]-1-fol))*mat.shape[1]])
			if z.shape[1] == left.shape[1]:
				left[i,:] = z
			else:
				left[i,:] = np.concatenate((np.reshape(mat[(i+1):,:], (1, ( fol - (i-(mat.shape[0]-1-fol)))*mat.shape[1])), z), axis = 1)
		else:
			left[i,:] = np.reshape(mat[(i+1) : (i+1+fol),:], (1, mat.shape[1]*fol))
	final = np.concatenate((right, mat, left), axis = 1)
	
	return final


### DATA READING AND SPLITTING ### (75% for training and 25% for testing)
###### Either replace this section with data extraction code or extract data beforehand and store them into readable files ######
print("Data reading and splitting")
files = [f for f in listdir(path + neutral_dir) if f.endswith(file_extension) and isfile(join(path + neutral_dir, f))]

ind_seq = list(range(len(files)))
shuffle(ind_seq)

## Shuffle the data
training_files = [files[i] for i in ind_seq]
testing_files = [files[i] for i in ind_seq]

## Split the data 
training_files = [training_files[i] for i in range(0,round(len(training_files)*90/100))] 
testing_files = [testing_files[i] for i in range(round(len(testing_files)*90/100), len(testing_files))]

# print("Concatenating training features") #without context
# training_mat_input = concat_to_mat(path + neutral_dir, training_files)
# training_mat_output = concat_to_mat(path + smile_dir, training_files)

print("Concatenating training features") #with context
training_mat_input = concat_to_mat_with_context(path + neutral_dir, training_files, prev, fol)
training_mat_output = concat_to_mat(path + smile_dir, training_files)

# print("Concatenating testing features")
# testing_mat_input = concat_to_mat(path + neutral_dir, testing_files)
# testing_mat_output = concat_to_mat(path + smile_dir, testing_files)

##DNN PART###
##Add/remove variables and parameters to change DNN architecture
n_hidden_1 = 257
n_hidden_2 = 125
n_hidden_3 = 75
n_hidden_4 = 50
n_hidden_5 = 75
n_hidden_6 = 125
n_hidden_7 = 257
# n_hidden_8 = 50
# n_hidden_9 = 75
# n_hidden_10 = 100
# n_hidden_11 = 125
n_input = training_mat_input.shape[1]
n_output = training_mat_output.shape[1]

X = tf.placeholder(tf.float32, shape = [None, n_input]) #input features
Y = tf.placeholder(tf.float32, shape = [None, n_output]) #output



weights = {
	'h1':tf.Variable(tf.random_normal([n_input, n_hidden_1])),
	'h2':tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
	'h3':tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
	'h4':tf.Variable(tf.random_normal([n_hidden_3, n_hidden_4])),
	'h5':tf.Variable(tf.random_normal([n_hidden_4, n_hidden_5])),
	'h6':tf.Variable(tf.random_normal([n_hidden_5, n_hidden_6])),
	'h7':tf.Variable(tf.random_normal([n_hidden_6, n_hidden_7])),
	# 'h8':tf.Variable(tf.random_normal([n_hidden_7, n_hidden_8])),
	# 'h9':tf.Variable(tf.random_normal([n_hidden_8, n_hidden_9])),
	# 'h10':tf.Variable(tf.random_normal([n_hidden_9, n_hidden_10])),
	# 'h11':tf.Variable(tf.random_normal([n_hidden_10, n_hidden_11])),
	'hout':tf.Variable(tf.random_normal([n_hidden_7, n_output]))
	
}

biases = {
	'b1':tf.Variable(tf.random_normal([n_hidden_1])),
	'b2':tf.Variable(tf.random_normal([n_hidden_2])),
	'b3':tf.Variable(tf.random_normal([n_hidden_3])),
	'b4':tf.Variable(tf.random_normal([n_hidden_4])),
	'b5':tf.Variable(tf.random_normal([n_hidden_5])),
	'b6':tf.Variable(tf.random_normal([n_hidden_6])),
	'b7':tf.Variable(tf.random_normal([n_hidden_7])),
	# 'b8':tf.Variable(tf.random_normal([n_hidden_8])),
	# 'b9':tf.Variable(tf.random_normal([n_hidden_9])),
	# 'b10':tf.Variable(tf.random_normal([n_hidden_10])),
	# 'b11':tf.Variable(tf.random_normal([n_hidden_11])),
	'bout':tf.Variable(tf.random_normal([n_output]))
}

### CREATE NETWORK ###
def create_network(x):
	l1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['h1']), biases['b1']))
	l2 = tf.nn.sigmoid(tf.add(tf.matmul(l1, weights['h2']), biases['b2']))
	l3 = tf.nn.sigmoid(tf.add(tf.matmul(l2, weights['h3']), biases['b3']))
	l4 = tf.nn.sigmoid(tf.add(tf.matmul(l3, weights['h4']), biases['b4']))
	l5 = tf.nn.sigmoid(tf.add(tf.matmul(l4, weights['h5']), biases['b5']))
	l6 = tf.nn.sigmoid(tf.add(tf.matmul(l5, weights['h6']), biases['b6']))
	l7 = tf.nn.sigmoid(tf.add(tf.matmul(l6, weights['h7']), biases['b7']))
	# l8 = tf.nn.sigmoid(tf.add(tf.matmul(l7, weights['h8']), biases['b8']))
	# l9 = tf.nn.sigmoid(tf.add(tf.matmul(l8, weights['h9']), biases['b9']))
	# l10 = tf.nn.sigmoid(tf.add(tf.matmul(l9, weights['h10']), biases['b10']))
	# l11 = tf.nn.sigmoid(tf.add(tf.matmul(l10, weights['h11']), biases['b11']))
	lout = tf.add(tf.matmul(l7, weights['hout']), biases['bout'])
	return lout

##With shared weights
def create_network_shared_weights(x):
	l1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['h1']), biases['b1']))
	l2 = tf.nn.sigmoid(tf.add(tf.matmul(l1, weights['h2']), biases['b2']))
	l3 = tf.nn.sigmoid(tf.add(tf.matmul(l2, weights['h3']), biases['b3']))
	l4 = tf.nn.sigmoid(tf.add(tf.matmul(l3, weights['h4']), biases['b4']))
	l5 = tf.nn.sigmoid(tf.add(tf.matmul(l4, tf.transpose(weights['h4'])), biases['b3']))
	l6 = tf.nn.sigmoid(tf.add(tf.matmul(l5, tf.transpose(weights['h3'])), biases['b2']))
	l7 = tf.nn.sigmoid(tf.add(tf.matmul(l6, tf.transpose(weights['h2'])), biases['b1']))
	# l8 = tf.nn.sigmoid(tf.add(tf.matmul(l7, weights['h8']), biases['b8']))
	lout = tf.add(tf.matmul(l7, weights['hout']), biases['bout'])
	return lout


##Build graph
net_op = create_network(X)



y_pred = net_op
y_real = Y

##Add cost function to graph
cost = tf.reduce_mean(tf.pow(y_pred - y_real, 2)) #euclidean distance
#cost = tf.reduce_mean(-tf.reduce_sum(tf.mul(y_real, tf.log(tf.clip_by_value(y_pred,1e-10,1.0))), reduction_indices=[1])) #cross-entropy

##Calculate and apply gradients in the same step

# optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

#create initializer for all variables
init = tf.initialize_all_variables()

with tf.Session() as sess:
	sess.run(init)
	#Training
	total_n_batch = int(len(training_files)/batch_size)
	for epoch in range(epochs):
		for i in range(total_n_batch):#mini batch training
			batch_in = [training_mat_input[k] for k in range(i*batch_size, (i+1)*batch_size)]
			batch_out = [training_mat_output[k] for k in range(i*batch_size, (i+1)*batch_size)]
			_, c = sess.run([optimizer, cost], feed_dict = {X: batch_in, Y: batch_out})
		if epoch %display_step == 0:
			print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c))
	print("Training Finished!!!")

	#Testing
	write_results_dir = path + "predicted_files/"
	batch_ts = testing_files[:number_of_tests] #take "number_of_tests" sentences for testing
	for ts_f in batch_ts:
		# ts_mat = read_list_float_txt_list(path + neutral_dir + ts_f) #if context is NOT taken into account
		ts_mat = concat_to_mat_with_context(path + neutral_dir, [ts_f], prev, fol) #if context is taken into account
		
		enc_dec = sess.run(y_pred, feed_dict={X: ts_mat})
		#enc_dec = sess.run(y_pred, feed_dict = {X: batch_ts, keep_prob: 1}) #with dropout
		enc_dec = np.array(enc_dec)

	#store results
		if not os.path.exists(write_results_dir):
			os.makedirs(write_results_dir)
		np.savetxt(write_results_dir + ts_f, enc_dec, delimiter = "\t", fmt='%7.6f')
		copyfile(path + neutral_dir + ts_f[:-len(file_extension)] + others_ext, write_results_dir + ts_f[:-len(file_extension)] + others_ext)

# #Write results in text files
# np.savetxt(write_results_dir + "predicted", enc_dec, delimiter = " ")
# np.savetxt(write_results_dir + "tests", testing_mat_output[:number_of_tests], delimiter = " ")
