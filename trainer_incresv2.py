import os
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
from optparse import OptionParser
import datetime as dt

# Import FCN Model
from inception_resnet_v2 import *
from pathlib import Path

inc_res_v2_checkpoint_file = './inception_resnet_v2_2016_08_30.ckpt'
my_file = Path(inc_res_v2_checkpoint_file)
if not my_file.is_file():
	# Download file from the link
	import wget
	import tarfile
	url = 'http://download.tensorflow.org/models/inception_resnet_v2_2016_08_30.tar.gz'
	filename = wget.download(url)

	# Extract the tar file
	tar = tarfile.open(filename)
	tar.extractall()
	tar.close()

# Command line options
parser = OptionParser()

# General settings
parser.add_option("-t", "--trainModel", action="store_true", dest="trainModel", default=False, help="Train model")
parser.add_option("-c", "--testModel", action="store_true", dest="testModel", default=False, help="Test model")
parser.add_option("-s", "--startTrainingFromScratch", action="store_true", dest="startTrainingFromScratch", default=False, help="Start training from scratch")
parser.add_option("-v", "--verbose", action="store", type="int", dest="verbose", default=0, help="Verbosity level")
parser.add_option("--tensorboardVisualization", action="store_true", dest="tensorboardVisualization", default=False, help="Enable tensorboard visualization")

# Input Reader Params
parser.add_option("--trainFileName", action="store", type="string", dest="trainFileName", default="train.idl", help="IDL file name for training")
parser.add_option("--testFileName", action="store", type="string", dest="testFileName", default="test.idl", help="IDL file name for testing")
parser.add_option("--imageWidth", action="store", type="int", dest="imageWidth", default=299, help="Image width for feeding into the network")
parser.add_option("--imageHeight", action="store", type="int", dest="imageHeight", default=299, help="Image height for feeding into the network")
parser.add_option("--imageChannels", action="store", type="int", dest="imageChannels", default=3, help="Number of channels in image for feeding into the network")
parser.add_option("--sequentialFetch", action="store_true", dest="sequentialFetch", default=False, help="Sequentially fetch images for each batch")
parser.add_option("--randomFetchTest", action="store_true", dest="randomFetchTest", default=False, help="Randomly fetch images for each test batch")

# Trainer Params
parser.add_option("--learningRate", action="store", type="float", dest="learningRate", default=1e-4, help="Learning rate")
parser.add_option("--trainingEpochs", action="store", type="int", dest="trainingEpochs", default=5, help="Training epochs")
parser.add_option("--batchSize", action="store", type="int", dest="batchSize", default=2, help="Batch size")
parser.add_option("--displayStep", action="store", type="int", dest="displayStep", default=5, help="Progress display step")
parser.add_option("--saveStep", action="store", type="int", dest="saveStep", default=1000, help="Progress save step")
parser.add_option("--evaluateStep", action="store", type="int", dest="evaluateStep", default=100000, help="Progress evaluation step")

# Directories
parser.add_option("--logsDir", action="store", type="string", dest="logsDir", default="./logs", help="Directory for saving logs")
parser.add_option("--modelDir", action="store", type="string", dest="modelDir", default="./model-inc_res_v2/", help="Directory for saving the model")
parser.add_option("--modelName", action="store", type="string", dest="modelName", default="inc_res_v2_fcn", help="Name to be used for saving the model")

# Network Params
parser.add_option("--numClasses", action="store", type="int", dest="numClasses", default=6, help="Number of classes")
parser.add_option("--neuronAliveProbability", action="store", type="float", dest="neuronAliveProbability", default=0.5, help="Probability of keeping a neuron active during training")

# Parse command line options
(options, args) = parser.parse_args()
print (options)

# Import custom data
import inputReader
inputReader = inputReader.InputReader(options)

if options.trainModel:
	with tf.variable_scope('Model'):
		# Data placeholders
		inputBatchImages = tf.placeholder(dtype=tf.float32, shape=[None, options.imageHeight,
											options.imageWidth, options.imageChannels], name="inputBatchImages")
		inputBatchLabels = tf.placeholder(dtype=tf.float32, shape=[None, options.numClasses], name="inputBatchLabels")
		inputKeepProbability = tf.placeholder(dtype=tf.float32, name="inputKeepProbability")

		scaledInputBatchImages = tf.scalar_mul((1.0/255), inputBatchImages)
		scaledInputBatchImages = tf.subtract(scaledInputBatchImages, 0.5)
		scaledInputBatchImages = tf.multiply(scaledInputBatchImages, 2.0)

	# Create model
	arg_scope = inception_resnet_v2_arg_scope()
	with slim.arg_scope(arg_scope):
		logits, end_points = inception_resnet_v2(scaledInputBatchImages, inputKeepProbability, options.numClasses, is_training=True)

	# Create list of vars to restore before train op
	variables_to_restore = slim.get_variables_to_restore(include=["InceptionResnetV2"])

	with tf.name_scope('Loss'):
		# Define loss
		# Reversed from slim.losses.softmax_cross_entropy(logits, labels) => tf.losses.softmax_cross_entropy(labels, logits)
		cross_entropy_loss = tf.losses.softmax_cross_entropy(inputBatchLabels, logits)
		# loss = tf.reduce_sum(slim.losses.get_regularization_losses()) + cross_entropy_loss

		tf.add_to_collection('losses', cross_entropy_loss)
		loss = tf.add_n(tf.get_collection('losses'), name='total_loss')

	with tf.name_scope('Accuracy'):
		correct_predictions = tf.equal(tf.argmax(end_points['Predictions'], 1), tf.argmax(inputBatchLabels, 1))
		accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name='accuracy')

	with tf.name_scope('Optimizer'):
		# Define Optimizer
		optimizer = tf.train.AdamOptimizer(learning_rate=options.learningRate)

		# Op to calculate every variable gradient
		gradients = tf.gradients(loss, tf.trainable_variables())
		gradients = list(zip(gradients, tf.trainable_variables()))

		# Op to update all variables according to their gradient
		applyGradients = optimizer.apply_gradients(grads_and_vars=gradients)

	# Initializing the variables
	init = tf.global_variables_initializer() # TensorFlow v0.11
	init_local = tf.local_variables_initializer()

	if options.tensorboardVisualization:
		# Create a summary to monitor cost tensor
		tf.summary.scalar("loss", loss)

		# Create summaries to visualize weights
		for var in tf.trainable_variables():
		    tf.summary.histogram(var.name, var)
		# Summarize all gradients
		for grad, var in gradients:
		    tf.summary.histogram(var.name + '/gradient', grad)

		# Merge all summaries into a single op
		mergedSummaryOp = tf.summary.merge_all()

	# 'Saver' op to save and restore all the variables
	saver = tf.train.Saver()

	bestLoss = 1e9
	step = 1

# Train model
if options.trainModel:
	with tf.Session() as sess:
		# Initialize all variables
		sess.run(init)
		sess.run(init_local)

		if options.startTrainingFromScratch:
			print ("Removing previous checkpoints and logs")
			os.system("rm -rf " + options.logsDir)
			os.system("rm -rf " + options.modelDir)
			os.system("mkdir " + options.modelDir)

			# Load the pre-trained Inception ResNet v2 model
			restorer = tf.train.Saver(variables_to_restore)
			restorer.restore(sess, inc_res_v2_checkpoint_file)

		# Restore checkpoint
		else:
			print ("Restoring from checkpoint")
			saver = tf.train.import_meta_graph(options.modelDir + options.modelName + ".meta")
			saver.restore(sess, options.modelDir + options.modelName)

		if options.tensorboardVisualization:
			# Op for writing logs to Tensorboard
			summaryWriter = tf.summary.FileWriter(options.logsDir, graph=tf.get_default_graph())

		print ("Starting network training")

		# Keep training until reach max iterations
		while True:
			batchImagesTrain, batchLabelsTrain = inputReader.getTrainBatch()
			# print ("Batch images shape: %s, Batch labels shape: %s" % (batchImagesTrain.shape, batchLabelsTrain.shape))

			# If training iterations completed
			if batchImagesTrain is None:
				print ("Training completed")
				break

			# Run optimization op (backprop)
			if options.tensorboardVisualization:
				[trainLoss, currentAcc, _, summary] = sess.run([loss, accuracy, applyGradients, mergedSummaryOp], feed_dict={inputBatchImages: batchImagesTrain, inputBatchLabels: batchLabelsTrain, inputKeepProbability: options.neuronAliveProbability})
				# Write logs at every iteration
				summaryWriter.add_summary(summary, step)
			else:
				[trainLoss, currentAcc, _] = sess.run([loss, accuracy, applyGradients], feed_dict={inputBatchImages: batchImagesTrain, inputBatchLabels: batchLabelsTrain, inputKeepProbability: options.neuronAliveProbability})

			print ("Iteration: %d, Minibatch Loss: %f, Accuracy: %f" % (step, trainLoss, currentAcc * 100))
			step += 1

			if step % options.saveStep == 0:
				# Save model weights to disk
				saver.save(sess, options.modelDir + options.modelName)
				print ("Model saved: %s" % (options.modelDir + options.modelName))

			#Check the accuracy on test data
			if step % options.evaluateStep == 0:
				# Report loss on test data
				batchImagesTest, batchLabelsTest = inputReader.getTestBatch()

				[testLoss, testAcc] = sess.run([loss, accuracy], feed_dict={inputBatchImages: batchImagesTest, inputBatchLabels: batchLabelsTest, inputKeepProbability: 1.0})
				print ("Test loss: %f, Test Accuracy: %f" % (testLoss, testAcc))

				# #Check the accuracy on test data
				# if step % options.saveStepBest == 0:
				# 	# Report loss on test data
				# 	batchImagesTest, batchLabelsTest = inputReader.getTestBatch()
				# 	[testLoss] = sess.run([loss], feed_dict={inputBatchImages: batchImagesTest, inputBatchLabels: batchLabelsTest, inputKeepProbability: 1.0})
				# 	print ("Test loss: %f" % testLoss)

				# 	# If its the best loss achieved so far, save the model
				# 	if testLoss < bestLoss:
				# 		bestLoss = testLoss
				# 		# bestModelSaver.save(sess, best_checkpoint_dir + 'checkpoint.data')
				# 		bestModelSaver.save(sess, checkpointPrefix, global_step=0, latest_filename=checkpointStateName)
				# 		print ("Best model saved in file: %s" % checkpointPrefix)
				# 	else:
				# 		print ("Previous best accuracy: %f" % bestLoss)

		# Save final model weights to disk
		saver.save(sess, options.modelDir + options.modelName)
		print ("Model saved: %s" % (options.modelDir + options.modelName))

		# Report loss on test data
		batchImagesTest, batchLabelsTest = inputReader.getTestBatch()
		testLoss = sess.run(loss, feed_dict={inputBatchImages: batchImagesTest, inputBatchLabels: batchLabelsTest, inputKeepProbability: 1.0})
		print ("Test loss (current): %f" % testLoss)

		print ("Optimization Finished!")

# Test model
if options.testModel:
	print ("Testing saved model")
	computeAccuracyWithoutOtherClass = False
	# Now we make sure the variable is now a constant, and that the graph still produces the expected result.
	with tf.Session() as session:
		saver = tf.train.import_meta_graph(options.modelDir + options.modelName + ".meta")
		saver.restore(session, options.modelDir + options.modelName)

		# Get reference to placeholders
		predictionsNode = session.graph.get_tensor_by_name("Predictions:0")
		accuracyNode = session.graph.get_tensor_by_name("Accuracy/accuracy:0")
		inputBatchImages = session.graph.get_tensor_by_name("Model/inputBatchImages:0")
		inputBatchLabels = session.graph.get_tensor_by_name("Model/inputBatchLabels:0")
		inputKeepProbability = session.graph.get_tensor_by_name("Model/inputKeepProbability:0")

		inputReader.resetTestBatchIndex()
		accumulatedAccuracy = 0.0
		numBatches = 0
		while True:
			extendDim = False if computeAccuracyWithoutOtherClass else True
			batchImagesTest, batchLabelsTest = inputReader.getTestBatch(extendDim=extendDim)
			if batchLabelsTest is None:
				break
			if computeAccuracyWithoutOtherClass:
				[predictions, accuracy] = session.run([predictionsNode, accuracyNode], feed_dict={inputBatchImages: batchImagesTest, inputBatchLabels: batchLabelsTest, inputKeepProbability: 1.0})
				print ('Current test batch accuracy (without other): %f' % (accuracy))
				accumulatedAccuracy += accuracy
			else:
				# Consider the prediction to be other class if the prediction confidence is less than 50%
				predictions = session.run(predictionsNode, feed_dict={inputBatchImages: batchImagesTest, inputKeepProbability: 1.0})
				predConf = np.max(predictions, axis=1)
				predClass = np.argmax(predictions, axis=1)
				actualClass = np.argmax(batchLabelsTest, axis=1)
				for i in range(predConf.shape[0]):
					print ("Prediction conf: %f, Predicted class: %d, GT: %d" % (predConf[i], predClass[i], actualClass[i]))
					if predConf[i] < 0.5:
						print ("(Low conf) Prediction conf: %f, Predicted class: %d, GT: %d" % (predConf[i], predClass[i], actualClass[i]))
						predClass[i] = options.numClasses # Others class
				accuracyWithOtherClass = np.mean(predClass == actualClass)
				print ('Current test batch accuracy (with other): %f' % (accuracyWithOtherClass))
				accumulatedAccuracy += accuracyWithOtherClass

			numBatches += 1

	accumulatedAccuracy = accumulatedAccuracy / numBatches
	print ('Cummulative test set accuracy: %f' % accumulatedAccuracy * 100)

	print ("Model tested")
