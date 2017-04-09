import os
import numpy as np
import skimage
import skimage.io
import skimage.transform

class InputReader:
	def __init__(self, options):
		self.options = options

		# Reads pathes of images together with their labels
		self.imageList, self.labelList = self.readImageNames(self.options.trainFileName)
		self.imageListTest, self.labelListTest = self.readImageNames(self.options.testFileName)

		# Shuffle the image list if sequential sampling is selected
		if self.options.sequentialFetch:
			np.random.shuffle(self.imageList)

		self.currentIndex = 0
		self.currentIndexTest = 0
		self.totalEpochs = 0
		self.totalImages = len(self.imageList)
		self.totalImagesTest = len(self.imageListTest)

		self.imgShape = [self.options.imageHeight, self.options.imageWidth, self.options.imageChannels]

	def readImageNames(self, imageListFile):
		"""Reads a .txt file containing paths and labels
		Args:
		   imageListFile: a .txt file with one /path/to/image per line along with their corresponding label
		Returns:
		   List with all fileNames in file imageListFile along with their corresponding label
		"""
		f = open(imageListFile, 'r')
		fileNames = []
		labels = []
		for line in f:
			data = line.strip().split(',')
			fileName = data[0].strip()
			label = int(data[1].strip())
			fileNames.append(fileName)
			labels.append(label)

		return fileNames, labels

	def readImagesFromDisk(self, fileNames):
		"""Consumes a list of filenames and returns images
		Args:
		  fileNames: List of image files
		Returns:
		  4-D numpy array: The input images
		"""
		images = []
		masks = []
		for i in range(0, len(fileNames)):			
			if self.options.verbose > 1:
				print ("Image: %s" % fileNames[i])

			# Read image
			img = skimage.io.imread(fileNames[i])
			
			if img.shape != self.imgShape:
				img = skimage.transform.resize(img, self.imgShape, preserve_range=True)
			images.append(img)

		# Convert list to ndarray
		images = np.array(images)

		return images

	def getTrainBatch(self):
		"""Returns training images and labels in batch
		Args:
		  None
		Returns:
		  Two numpy arrays: training images and labels in batch.
		"""
		if self.totalEpochs >= self.options.trainingEpochs:
			return None, None

		endIndex = self.currentIndex + self.options.batchSize
		if self.options.sequentialFetch:
			# Fetch the next sequence of images
			self.indices = np.arange(self.currentIndex, endIndex)

			if endIndex > self.totalImages:
				# Replace the indices which overshot with 0
				self.indices[self.indices >= self.totalImages] = np.arange(0, np.sum(self.indices >= self.totalImages))
		else:
			# Randomly fetch any images
			self.indices = np.random.choice(self.totalImages, self.options.batchSize)

		imagesBatch = self.readImagesFromDisk([self.imageList[index] for index in self.indices])
		labelsBatch = self.convertLabelsToOneHot([self.labelList[index] for index in self.indices])

		self.currentIndex = endIndex
		if self.currentIndex > self.totalImages:
			print ("Training epochs completed: %f" % (self.totalEpochs + (float(self.currentIndex) / self.totalImages)))
			self.currentIndex = self.currentIndex - self.totalImages
			self.totalEpochs = self.totalEpochs + 1

			# Shuffle the image list if not random sampling at each stage
			if self.options.sequentialFetch:
				np.random.shuffle(self.imageList)

		return imagesBatch, labelsBatch

	def resetTestBatchIndex(self):
		self.currentIndexTest = 0

	def getTestBatch(self, extendDim=False):
		"""Returns testing images and labels in batch
		Args:
		  None
		Returns:
		  Two numpy arrays: test images and labels in batch.
		"""
		# Optional Image and Label Batching
		if self.currentIndexTest >= self.totalImagesTest:
			return None, None

		if self.options.randomFetchTest:
			self.indices = np.random.choice(self.totalImagesTest, self.options.batchSize)
			imagesBatch = self.readImagesFromDisk([self.imageListTest[index] for index in self.indices])
			labelsBatch = self.convertLabelsToOneHot([self.labelList[index] for index in self.indices], extendDim=extendDim)
			
		else:
			endIndex = self.currentIndexTest + self.options.batchSize
			if endIndex > self.totalImagesTest:
				endIndex = self.totalImagesTest
			self.indices = np.arange(self.currentIndexTest, endIndex)
			imagesBatch = self.readImagesFromDisk([self.imageListTest[index] for index in self.indices])
			labelsBatch = self.convertLabelsToOneHot([self.labelListTest[index] for index in self.indices], extendDim=extendDim)
			self.currentIndexTest = endIndex

		return imagesBatch, labelsBatch

	def restoreCheckpoint(self, numSteps):
		"""Restores current index and epochs using numSteps
		Args:
		  numSteps: Number of batches processed
		Returns:
		  None
		"""
		processedImages = numSteps * self.options.batchSize
		self.totalEpochs = processedImages / self.totalImages
		self.currentIndex = processedImages % self.totalImages

	def convertLabelsToOneHot(self, labels, extendDim=False):
		oneHotLabels = []
		numElements = self.options.numClasses
		if extendDim:
				numElements += 1

		for label in labels:
			oneHotVector = np.zeros(numElements)
			oneHotVector[label] = 1
			oneHotLabels.append(oneHotVector)
		
		oneHotLabels = np.array(oneHotLabels)
		return oneHotLabels