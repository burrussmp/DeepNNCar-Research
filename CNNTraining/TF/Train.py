#Train.py
#Authored: Shreyas Ramakrishna and Matthew Burruss
#Last Edited: 03/28/2019
#Description: Model training in tensorflow, which saves the model and weights with lowest loss

import os
import tensorflow as tf
import time
import csv
import DataSetLoader
import model
import sys
import copy
import random
import glob

best_train_loss = 0.0
last_improvement = 0
i=0
batch_data_training = True
batch_size = 3000
csvfile = open("model.csv", "w")
#######Parameters########
save_path = "./model"
model_name = "test"
training_steps=200
model_load_file="test.ckpt"
#########################

##########Saving Directories##########
save_dir = os.path.abspath('model')
######################################

def load_list_of_datasets(array_of_csv_paths):
	images = []
	outputs = []
	for dataset in array_of_csv_paths:
		inputs = DataSetLoader.load_training_images(dataset)
		out= DataSetLoader.read_training_output_data(dataset)
		images = images + inputs
		outputs = outputs + out
	print("loaded datset of %d inputs and outputs" %len(outputs))
	return images,outputs

def batch(images,outputs,batch_size):
	n = len(images)
	value = random.sample(range(0, n), batch_size)
	batch_images = []
	batch_outputs = []
	for i in value:
		batch_images.append(images[i])
		batch_outputs.append(outputs[i])
	return batch_images,batch_outputs

def train(images_training,outputs_training,images_validation,outputs_validation):

	#if not os.path.isdir(out_dir):
	#   os.makedirs(out_dir)

	if not os.path.isdir(save_dir):
	    os.makedirs(save_dir)

	def join(dirpath, filename):
	    return os.path.join(dirpath, filename)

	sess = tf.InteractiveSession()
	loss=tf.losses.mean_squared_error(model.y_,model.y)
	#Use adam optimizer
	train_step = tf.train.AdamOptimizer(0.0001).minimize(loss)
	saver = tf.train.Saver()
	model_load_path = join(save_dir, model_load_file)
	print("....Loading existing model from.... %s" %model_load_path)

	if os.path.exists(model_load_path + ".index"):
	    print ("....Loading initial weights from %s...." % model_load_path)
	    saver.restore(sess, model_load_path)
	else:
	    print ("....Initialize weights....")
	init = tf.global_variables_initializer()

	sess.run(init)
	n = 0
	k = 0
	for i in range(training_steps):
		lossdata =[]
		#Unshuffled Full data training and validation
		validate = False
		print("Reloading input/outputs")
		if((i+1) % 10)!=0:
			print("...............training step {}...............".format(i+1))
			#inputs = DataSetLoader.load_training_images(training_dataset)
			#outputs= DataSetLoader.read_training_output_data(training_dataset)
			
			# make deep copy of training inputs/outputs
			inputs = copy.deepcopy(images_training)
			outputs = copy.deepcopy(outputs_training)


		if((i+1) % 10)==0:
			print("...............validation step {}...............".format(i+1))
			#inputs1 = DataSetLoader.load_validation_images(validation_dataset)
			#outputs1 = DataSetLoader.read_validation_output_data(validation_dataset)

			# made deeep copy of validation inputs/outputs
			inputs1 = copy.deepcopy(images_validation)
			outputs1 = copy.deepcopy(outputs_validation)
			validate = True
		#Shuffled batch data for training and validation
		if(batch_data_training == True):
			print("Batching size: %d" %batch_size)
			if((i+1) % 10)!=0:
				print("...............training step {}...............".format(i+1))
				#inputs,outputs = DataSetLoader.load_images_and_outputs_batch(training_dataset,batch_size)

				# select random batch from images for training
				inputs,outputs = batch(inputs,outputs,batch_size)

			if((i+1) % 10)==0:
				print("...............validation step {}...............".format(i+1))
				#inputs1,outputs1 = DataSetLoader.load_images_and_outputs_batch(validation_dataset,batch_size)
				validate = True
				inputs1,outputs1 = batch(inputs1,outputs1,batch_size)
				# select random batch from images for validation
		
       		 #Running the model with the data
		if (not validate):
			train_step.run(feed_dict={model.x: inputs, model.y_:outputs})
			t_loss = loss.eval(feed_dict={model.x: inputs, model.y_:outputs})
			print("step {} of {},train loss {}".format(i+1, training_steps, t_loss))
		else:
			v_loss = loss.eval(feed_dict={model.x: inputs1, model.y_: outputs1})
			print("step {} of {},validation loss {}".format(i+1, training_steps, v_loss))
		
		if((i+1) % 10)!= 0:
			#getting the training loss
			t_loss = loss.eval(feed_dict={model.x: inputs, model.y_: outputs})
			print("step {} of {},train loss {}".format(i+1, training_steps, t_loss))
			writer = csv.writer(csvfile)
			writer.writerow([t_loss])

		if((i+1) % 10) == 0:

  			#t_loss = loss.eval(feed_dict={model.x: inputs, model.y_: outputs})
			v_loss = loss.eval(feed_dict={model.x: inputs, model.y_: outputs})
			print("step {} of {},validation loss {}".format(i+1, training_steps, v_loss))
			writer = csv.writer(csvfile)
			writer.writerow([v_loss])
		
		if(i==0):
		    best_train_loss = t_loss
		    saver.save(sess, save_path + '/' + model_name + '.ckpt')
		    print("Saving the first model with name {} for training step{}:".format(model_name, i+1))

		if(i>0 and t_loss < best_train_loss):
			k = 0
			epoch_num=i
			#loss = (best_train_loss - t_loss)
			print("loss has improved by %f"%(best_train_loss - t_loss))
			saver.save(sess, save_path + '/' + model_name + '.ckpt')
			print("Saving the best model with name {} for training step{}:".format(model_name, i+1))
			best_train_loss = t_loss

		if(i>0 and t_loss > best_train_loss):
			k = k+1#counter for not entering the overfitting phase
			if(k==5):
				assert (k == 5),"Entering the overfitting phase!"

		# appending loss data for plotting
		if (not validate):
			lossdata.append(t_loss)
		else:
			lossdata.append(v_loss)
		writer = csv.writer(csvfile)
		writer.writerow(lossdata)

	print("The best model was saved at epoch {} and has a loss of {}".format(epoch_num+1, best_train_loss))
	time.sleep(20)


if __name__ == '__main__':
	path = "C:\\Users\\Matthew Burruss\\Documents\\GitHub\DeepNNCar Public\\CNNTraining\\TF\\"
	d1 = []
	d1.append(path+"Data20190401042944500.csv")
	#d1.append(path+"Data20190401041323132.csv")
	#d1.append(path+"Data20190401035027367.csv")
	#d1.append(path+"Data20190401035027367.csv")
	#d1.append(path+"Data20190401032355775.csv")
	print("loading training dataset")
	images_training,outputs_training = load_list_of_datasets(d1)
	# possible add validation dataset
	print("loading validation dataset")
	images_validation = copy.deepcopy(images_training)
	outputs_validation = copy.deepcopy(outputs_training)
	train(images_training,outputs_training,images_validation,outputs_validation)
