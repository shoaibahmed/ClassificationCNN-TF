import tensorflow as tf
slim = tf.contrib.slim

import numpy as np

from optparse import OptionParser
import wget
import tarfile
import os
import cv2
import time
import shutil

import inception_resnet_v2
import resnet_v1
import nasnet.nasnet as nasnet

from sklearn import svm

TRAIN = 0
VAL = 1
TEST = 2

import sys

if sys.version_info[0] == 3:
	print ("Using Python 3")
	import pickle as cPickle
else:
	print ("Using Python 2")
	import cPickle

# Command line options
parser = OptionParser()

# General settings
parser.add_option("-m", "--model", action="store", type="string", dest="model", default="NAS", help="Model to be used for Cross-Layer Pooling")
parser.add_option("-t", "--trainModel", action="store_true", dest="trainModel", default=False, help="Train model")
parser.add_option("-c", "--testModel", action="store_true", dest="testModel", default=False, help="Test model")
parser.add_option("-s", "--startTrainingFromScratch", action="store_true", dest="startTrainingFromScratch", default=False, help="Start training from scratch")
parser.add_option("-v", "--tensorboardVisualization", action="store_true", dest="tensorboardVisualization", default=False, help="Enable tensorboard visualization")

# Input Reader Params
parser.add_option("--imageWidth", action="store", type="int", dest="imageWidth", default=224, help="Image width for feeding into the network")
parser.add_option("--imageHeight", action="store", type="int", dest="imageHeight", default=224, help="Image height for feeding into the network")
parser.add_option("--imageChannels", action="store", type="int", dest="imageChannels", default=3, help="Number of channels in the image")
parser.add_option("--resizeRatio", action="store", type="float", dest="resizeRatio", default=1.15, help="Resizing image ratio")
parser.add_option("--useImageMean", action="store_true", dest="useImageMean", default=False, help="Use image mean for normalization")

# Trainer Params
parser.add_option("--learningRate", action="store", type="float", dest="learningRate", default=1e-4, help="Learning rate")
parser.add_option("--labelSmoothing", action="store", type="float", dest="labelSmoothing", default=0.1, help="Label smoothing parameter")
parser.add_option("--weightedSoftmax", action="store_true", dest="weightedSoftmax", default=False, help="Use weighted softmax")
parser.add_option("--trainingEpochs", action="store", type="int", dest="trainingEpochs", default=10, help="Training epochs")
parser.add_option("--batchSize", action="store", type="int", dest="batchSize", default=5, help="Batch size")
parser.add_option("--saveStep", action="store", type="int", dest="saveStep", default=1000, help="Progress save step")
parser.add_option("--displayStep", action="store", type="int", dest="displayStep", default=5, help="Progress display step")
parser.add_option("--trainSVM", action="store_true", dest="trainSVM", default=False, help="Train SVM on top of the features extracted from the trained model")

# Directories
parser.add_option("--logsDir", action="store", type="string", dest="logsDir", default="./logs", help="Directory for saving logs")
parser.add_option("--modelDir", action="store", type="string", dest="modelDir", default="./mymodel/", help="Directory for saving the model")
parser.add_option("--modelName", action="store", type="string", dest="modelName", default="mymodel", help="Name to be used for saving the model")

parser.add_option("--trainDataFile", action="store", type="string", dest="trainDataFile", default="/netscratch/siddiqui/CrossLayerPooling/data/opticdisc-labels-train.txt", help="Training data file")
parser.add_option("--valDataFile", action="store", type="string", dest="valDataFile", default="/netscratch/siddiqui/CrossLayerPooling/data/opticdisc-labels-val.txt", help="Validation data file")
parser.add_option("--testDataFile", action="store", type="string", dest="testDataFile", default="/netscratch/siddiqui/CrossLayerPooling/data/opticdisc-labels-test.txt", help="Test data file")

# Parse command line options
(options, args) = parser.parse_args()
print (options)

baseDir = os.getcwd()

# Load the model
if options.model == "ResNet":
	resnet_checkpoint_file = os.path.join(baseDir, 'resnet_v1_152.ckpt')
	if not os.path.isfile(resnet_checkpoint_file):
		# Download file from the link
		url = 'http://download.tensorflow.org/models/resnet_v1_152_2016_08_28.tar.gz'
		filename = wget.download(url)

		# Extract the tar file
		tar = tarfile.open(filename)
		tar.extractall()
		tar.close()

	options.imageHeight = options.imageWidth = 224

elif options.model == "IncResV2":
	inc_res_v2_checkpoint_file = os.path.join(baseDir, 'inception_resnet_v2_2016_08_30.ckpt')
	if not os.path.isfile(inc_res_v2_checkpoint_file):
		# Download file from the link
		url = 'http://download.tensorflow.org/models/inception_resnet_v2_2016_08_30.tar.gz'
		filename = wget.download(url)

		# Extract the tar file
		tar = tarfile.open(filename)
		tar.extractall()
		tar.close()

	options.imageHeight = options.imageWidth = 299

elif options.model == "NAS": 
	nas_checkpoint_file = os.path.join(baseDir, 'model.ckpt')
	if not os.path.isfile(nas_checkpoint_file + '.index'):
		# Download file from the link
		url = 'https://storage.googleapis.com/download.tensorflow.org/models/nasnet-a_large_04_10_2017.tar.gz'
		filename = wget.download(url)

		# Extract the tar file
		tar = tarfile.open(filename)
		tar.extractall()
		tar.close()

	# Update image sizes
	options.imageHeight = options.imageWidth = 331

else:
	print ("Error: Unknown model selected")
	exit(-1)

# Define params
IMAGENET_MEAN = [123.68, 116.779, 103.939] # RGB

# Decide the resizing dimensions
RESIZED_IMAGE_DIMS = [int(options.imageHeight * options.resizeRatio), int(options.imageWidth * options.resizeRatio)]
print ("Resized image dimensions: %s" % str(RESIZED_IMAGE_DIMS))

# Reads an image from a file, decodes it into a dense tensor
def _parse_function(filename, label):
	image_string = tf.read_file(filename)
	img = tf.image.decode_jpeg(image_string)

	if options.trainModel:
		img = tf.image.resize_images(img, RESIZED_IMAGE_DIMS)

		# Random crop
		img = tf.random_crop(img, [options.imageHeight, options.imageWidth, options.imageChannels])

		# Random flipping
		img = tf.image.random_flip_left_right(img)
		img = tf.image.random_flip_up_down(img)

		# Random image distortions
		# img = tf.image.random_brightness(img, max_delta=0.1)

	else:
		img = tf.image.resize_images(img, [options.imageHeight, options.imageWidth])

	img.set_shape([options.imageHeight, options.imageWidth, options.imageChannels])
	img = tf.cast(img, tf.float32) # Convert to float tensor
	return img, filename, label

def loadDataset(currentDataFile):
	print ("Loading data from file: %s" % (currentDataFile))
	dataClasses = {}
	with open(currentDataFile) as f:
		imagesBaseDir = '/netscratch/siddiqui/CrossLayerPooling/data/OpticDiscs/'
		imageFileNames = f.readlines()
		imNames = []
		imLabels = []
		for imName in imageFileNames:
			imName = imName.strip().split(' ')
			imNames.append(os.path.join(imagesBaseDir, imName[0]))
			imLabels.append(int(imName[1]))

			if int(imName[1]) not in dataClasses:
				dataClasses[int(imName[1])] = 1
			else:
				dataClasses[int(imName[1])] += 1

		# imageFileNames = [x.strip().split(' ') for x in imageFileNames] # FileName and Label is separated by a space
		imNames = tf.constant(imNames)
		imLabels = tf.constant(imLabels)

	numClasses = len(dataClasses)
	numFiles = len(imageFileNames)
	print ("Dataset loaded")
	print ("Files: %d | Classes: %d" % (numFiles, numClasses))
	print (dataClasses)
	classWeights = [float(numFiles - dataClasses[x]) / float(numFiles) for x in dataClasses]
	print ("Class weights: %s" % str(classWeights))

	dataset = tf.contrib.data.Dataset.from_tensor_slices((imNames, imLabels))
	dataset = dataset.map(_parse_function)
	dataset = dataset.shuffle(buffer_size=numFiles)
	dataset = dataset.batch(options.batchSize)

	return dataset, numClasses, classWeights

# A vector of filenames
trainDataset, numClasses, classWeights = loadDataset(options.trainDataFile)
valDataset, _, _ = loadDataset(options.valDataFile)
testDataset, _, _ = loadDataset(options.testDataFile)

trainIterator = trainDataset.make_initializable_iterator()
valIterator = valDataset.make_initializable_iterator()
testIterator = testDataset.make_initializable_iterator()

global_step = tf.train.get_or_create_global_step()

with tf.name_scope('Model'):
	# Data placeholders
	datasetSelectionPlaceholder = tf.placeholder(dtype=tf.int32, shape=(), name='DatasetSelectionPlaceholder')
	inputBatchImages, inputBatchImageNames, inputBatchLabels = tf.cond(tf.equal(datasetSelectionPlaceholder, TRAIN), lambda: trainIterator.get_next(), 
																lambda: tf.cond(tf.equal(datasetSelectionPlaceholder, VAL), lambda: valIterator.get_next(), lambda: testIterator.get_next()))
	inputBatchImageLabels = tf.one_hot(inputBatchLabels, depth=numClasses)

	print ("Data shape: %s" % str(inputBatchImages.get_shape()))
	print ("Labels shape: %s" % str(inputBatchImageLabels.get_shape()))

	if options.model == "IncResV2":
		scaledInputBatchImages = tf.scalar_mul((1.0 / 255.0), inputBatchImages)
		scaledInputBatchImages = tf.subtract(scaledInputBatchImages, 0.5)
		scaledInputBatchImages = tf.multiply(scaledInputBatchImages, 2.0)

		# Create model
		arg_scope = inception_resnet_v2.inception_resnet_v2_arg_scope()
		with slim.arg_scope(arg_scope):
			# logits, end_points = inception_resnet_v2.inception_resnet_v2(scaledInputBatchImages, is_training=options.trainModel, num_classes=numClasses)
			# logits, end_points = inception_resnet_v2.inception_resnet_v2(scaledInputBatchImages, is_training=True, dropout_keep_prob=0.5 if options.trainModel else 1.0, num_classes=numClasses)
			logits, end_points = inception_resnet_v2.inception_resnet_v2(scaledInputBatchImages, is_training=False, num_classes=numClasses)

		# Create list of vars to restore before train op (exclude the logits due to change in number of classes)
		variables_to_restore = slim.get_variables_to_restore(exclude=["InceptionResnetV2/Logits", "InceptionResnetV2/AuxLogits"])

	elif options.model == "ResNet":
		if options.useImageMean:
			imageMean = tf.reduce_mean(inputBatchImages, axis=[1, 2], keep_dims=True)
			print ("Image mean shape: %s" % str(imageMean.shape))
			processedInputBatchImages = inputBatchImages - imageMean
		else:
			print (inputBatchImages.shape)
			channels = tf.split(axis=3, num_or_size_splits=options.imageChannels, value=inputBatchImages)
			for i in range(options.imageChannels):
				channels[i] -= IMAGENET_MEAN[i]
			processedInputBatchImages = tf.concat(axis=3, values=channels)
			print (processedInputBatchImages.shape)

		# Create model
		arg_scope = resnet_v1.resnet_arg_scope()
		with slim.arg_scope(arg_scope):
			# logits, end_points = resnet_v1.resnet_v1_152(processedInputBatchImages, is_training=options.trainModel, num_classes=numClasses)
			logits, end_points = resnet_v1.resnet_v1_152(processedInputBatchImages, is_training=False, num_classes=numClasses)

		# Create list of vars to restore before train op (exclude the logits due to change in number of classes)
		variables_to_restore = slim.get_variables_to_restore(exclude=["resnet_v1_152/logits", "resnet_v1_152/AuxLogits"])

	elif options.model == "NAS":
		scaledInputBatchImages = tf.scalar_mul((1.0 / 255.0), inputBatchImages)
		scaledInputBatchImages = tf.subtract(scaledInputBatchImages, 0.5)
		scaledInputBatchImages = tf.multiply(scaledInputBatchImages, 2.0)

		# Create model
		arg_scope = nasnet.nasnet_large_arg_scope()
		with slim.arg_scope(arg_scope):
			# logits, end_points = nasnet.build_nasnet_large(scaledInputBatchImages, is_training=options.trainModel, is_batchnorm_training=options.trainModel, num_classes=numClasses)
			logits, end_points = nasnet.build_nasnet_large(scaledInputBatchImages, is_training=options.trainModel, is_batchnorm_training=False, num_classes=numClasses)

		# Create list of vars to restore before train op (exclude the logits due to change in number of classes)
		variables_to_restore = slim.get_variables_to_restore(exclude=["aux_11/aux_logits/FC", "final_layer/FC"])

	else:
		print ("Error: Unknown model selected")
		exit(-1)

with tf.name_scope('Loss'):
	if options.weightedSoftmax:
		print ("Using weighted cross-entropy loss")
		# Define the class weightages (weighted softmax)
		classWeightsTensor = tf.constant(classWeights)
		classWeights = tf.gather(classWeightsTensor, inputBatchLabels)
	else:
		print ("Using unweighted cross-entropy loss")
		classWeights = tf.ones_like(inputBatchLabels)

	# Define loss
	cross_entropy_loss = tf.losses.softmax_cross_entropy(onehot_labels=inputBatchImageLabels, logits=logits, weights=classWeights, label_smoothing=options.labelSmoothing)
	tf.losses.add_loss(cross_entropy_loss)
	loss = tf.reduce_mean(tf.losses.get_losses())

with tf.name_scope('Accuracy'):
	correct_predictions = tf.equal(tf.argmax(logits, 1), tf.argmax(inputBatchImageLabels, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name='accuracy')

with tf.name_scope('Optimizer'):
	# Define Optimizer
	# trainOp = tf.train.AdamOptimizer(learning_rate=options.learningRate).minimize(loss)
	optimizer = tf.train.AdamOptimizer(learning_rate=options.learningRate)

	# Op to calculate every variable gradient
	gradients = tf.gradients(loss, tf.trainable_variables())
	gradients = list(zip(gradients, tf.trainable_variables()))

	# Op to update all variables according to their gradient
	update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) # Added for batch-norm
	with tf.control_dependencies(update_ops):
		trainOp = optimizer.apply_gradients(grads_and_vars=gradients)

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
		if grad is not None:
			tf.summary.histogram(var.name + '/gradient', grad)

	# Merge all summaries into a single op
	mergedSummaryOp = tf.summary.merge_all()

# GPU config
config = tf.ConfigProto()
config.gpu_options.allow_growth=True

# Train model
if options.trainModel:
	with tf.Session(config=config) as sess:
		# Initialize all vars
		sess.run(init)
		sess.run(init_local)

		# Restore the model params
		if options.startTrainingFromScratch:
			print ("Removing previous checkpoints and logs")
			os.system("rm -rf " + options.logsDir)
			os.system("rm -rf " + options.modelDir)
			os.system("mkdir " + options.modelDir)

			checkpointFileName = resnet_checkpoint_file if options.model == "ResNet" else inc_res_v2_checkpoint_file if options.model == "IncResV2" else nas_checkpoint_file
			print ("Restoring weights from file: %s" % (checkpointFileName))

			# Load the imagenet pre-trained model
			restorer = tf.train.Saver(variables_to_restore)
			restorer.restore(sess, checkpointFileName)
		else:
			# Load the user trained model
			restorer = tf.train.Saver(variables_to_restore)
			restorer.restore(sess, os.path.join(options.modelDir, options.modelName))

		# Saver op to save and restore all the variables
		saver = tf.train.Saver()

		if options.tensorboardVisualization:
			# Write the graph to file
			summaryWriter = tf.summary.FileWriter(options.logsDir, graph=tf.get_default_graph())

		globalStep = 0
		numEpochs = options.trainingEpochs + 1 if options.trainSVM else options.trainingEpochs
		if options.trainSVM:
			imageNames = []
			imageLabels = []
			imageFeatures = []

		for epoch in range(numEpochs):
			# Initialize the dataset iterator
			sess.run(trainIterator.initializer)
			isLastEpoch = epoch == options.trainingEpochs
			try:
				step = 0
				while True:
					start_time = time.time()

					if isLastEpoch:
						# Collect features for SVM
						[imageName, imageLabel, featureVec] = sess.run([inputBatchImageNames, inputBatchLabels, end_points['global_pool']], feed_dict={datasetSelectionPlaceholder: TRAIN})
						imageNames.extend(imageName)
						imageLabels.extend(imageLabel)
						imageFeatures.extend(np.squeeze(featureVec))

						duration = time.time() - start_time

						# Print an overview fairly often.
						if step % options.displayStep == 0:
							print('Step: %d | Duration: %f' % (step, duration))
					else:
						# Run optimization op (backprop)
						if options.tensorboardVisualization:
							[trainLoss, currentAcc, _, summary] = sess.run([loss, accuracy, trainOp, mergedSummaryOp], feed_dict={datasetSelectionPlaceholder: TRAIN})
							summaryWriter.add_summary(summary, globalStep)
						else:
							[trainLoss, currentAcc, _] = sess.run([loss, accuracy, trainOp], feed_dict={datasetSelectionPlaceholder: TRAIN})

						duration = time.time() - start_time

						# Print an overview fairly often.
						if step % options.displayStep == 0:
							print('Step: %d | Loss: %f | Accuracy: %f | Duration: %f' % (step, trainLoss, currentAcc, duration))
					
					step += 1
					globalStep += 1

			except tf.errors.OutOfRangeError:
				print('Done training for %d epochs, %d steps.' % (epoch, step))

		# Save final model weights to disk
		saver.save(sess, os.path.join(options.modelDir, options.modelName))
		print ("Model saved: %s" % (os.path.join(options.modelDir, options.modelName)))

		if options.trainSVM:
			# Train the SVM
			print ("Training SVM")
			imageFeatures = np.array(imageFeatures)
			imageLabels = np.array(imageLabels)
			print ("Data shape: %s" % str(imageFeatures.shape))
			print ("Labels shape: %s" % str(imageLabels.shape))

			clf = svm.LinearSVC(C=1.0)
			clf.fit(imageFeatures, imageLabels)
			print ("Training Complete!")

			with open(os.path.join(options.modelDir, 'svm.pkl'), 'wb') as fid:
				cPickle.dump(clf, fid)

			print ("Evaluating performance on training data")
			trainAccuracy = clf.score(imageFeatures, imageLabels)
			print ("Train accuracy: %f" % (trainAccuracy))

	print ("Optimization Finished!")

# Test model
if options.testModel:
	print ("Testing saved model")

	# Now we make sure the variable is now a constant, and that the graph still produces the expected result.
	with tf.Session(config=config) as sess:
		# Saver op to save and restore all the variables
		saver = tf.train.Saver()
		saver.restore(sess, os.path.join(options.modelDir, options.modelName))

		# Initialize the dataset iterator
		sess.run(testIterator.initializer)

		svmFound = False
		if os.path.exists(os.path.join(options.modelDir, 'svm.pkl')):
			print ("Loading saved SVM instance")
			with open(os.path.join(options.modelDir, 'svm.pkl'), 'rb') as fid:
				clf = cPickle.load(fid)
				if clf is None:
					print ("Error: Unable to load SVM instance.")
					exit (-1)
				svmFound = True
				print ("SVM instance loaded successfully!")

		try:
			step = 0
			correctInstances = 0
			totalInstances = 0

			if svmFound:
				correctInstancesSVM = 0
				imageLabels = []
				imageFeatures = []

			while True:
				start_time = time.time()
				
				[batchLabelsTest, predictions, currentAcc, featureVec] = sess.run([inputBatchImageLabels, logits, accuracy, end_points['global_pool']], feed_dict={datasetSelectionPlaceholder: TEST})

				predConf = np.max(predictions, axis=1)
				predClass = np.argmax(predictions, axis=1)
				actualClass = np.argmax(batchLabelsTest, axis=1)

				correctInstances += np.sum(predClass == actualClass)
				totalInstances += predClass.shape[0]

				if svmFound:
					imageLabels.extend(actualClass)
					imageFeatures.extend(np.squeeze(featureVec))

				duration = time.time() - start_time
				print('Step: %d | Accuracy: %f | Duration: %f' % (step, currentAcc, duration))

				step += 1
		except tf.errors.OutOfRangeError:
			print('Done testing for %d epochs, %d steps.' % (1, step))

	print ('Number of test images: %d' % (totalInstances))
	print ('Number of correctly predicted images: %d' % (correctInstances))
	print ('Test set accuracy: %f' % ((float(correctInstances) / float(totalInstances)) * 100))

	if svmFound:
		print ("Evaluating SVM")
		imageFeatures = np.array(imageFeatures)
		imageLabels = np.array(imageLabels)
		print ("Data shape: %s" % str(imageFeatures.shape))
		print ("Labels shape: %s" % str(imageLabels.shape))

		testAccuracy = clf.score(imageFeatures, imageLabels)
		print ("Test accuracy: %f" % (testAccuracy))
