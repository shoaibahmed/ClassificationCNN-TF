import os
import os.path
import numpy as np
import skimage
import skimage.io
import skimage.transform
import skimage.color
import random

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
		self.imgShapeTrain = [350, 350, self.options.imageChannels]

		if options.computeMeanImage:
			meanFile = "meanImg.npy"
			if os.path.isfile(meanFile):
				self.meanImg = np.load(meanFile)
				print ("Image mean: %s" % str(self.meanImg))
			else:
				print ("Computing mean image from training dataset")
				# Compute mean image
				meanImg = np.zeros([options.imageHeight, options.imageWidth, options.imageChannels])
				imagesProcessed = 0
				for i in range(len(self.imageList)):
					try:
						img = skimage.io.imread(self.imageList[i])

						# Convert image to rgb if grayscale
						if len(img.shape) == 2:
							img = skimage.color.gray2rgb(img)

						if img.shape != self.imgShape:
							img = skimage.transform.resize(img, self.imgShape, preserve_range=True)

					except:
						print ("Unable to load image: %s" % self.imageList[i])
						continue
					
					img = img.astype(float)
					meanImg += img
					imagesProcessed += 1

				meanImg = meanImg / imagesProcessed
				self.meanImg = meanImg
				np.save("fullImageMean.npy", self.meanImg)

				# Convert mean to per channel mean (single channel images)
				self.meanImg = np.mean(np.mean(self.meanImg, axis=0), axis=0)
				np.save(meanFile, self.meanImg)

				print ("Mean image computed")

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
			data = line.strip().split(' ')
			fileName = data[0].strip()
			label = int(data[1].strip())
			fileNames.append(self.options.imagesBaseDir + fileName)
			labels.append(label)

		return fileNames, labels

	def readImagesFromDisk(self, fileNames, isTrain = True):
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

			# Convert image to rgb if grayscale
			if len(img.shape) == 2:
				img = skimage.color.gray2rgb(img)

			if isTrain:
				if img.shape != self.imgShapeTrain:
					img = skimage.transform.resize(img, self.imgShapeTrain, preserve_range=True)

				# Take a random crop from the image
				randY = random.randint(0, 350 - self.options.imageHeight - 1)
				randX = random.randint(0, 350 - self.options.imageWidth - 1)
				img = img[randY:(randY+self.options.imageHeight), randX:(randX+self.options.imageWidth)]

				if self.options.computeMeanImage:
					img = img - self.meanImg

				if random.random() > 0.5:
					img = np.fliplr(img)

			else:
				if img.shape != self.imgShape:
					img = skimage.transform.resize(img, self.imgShape, preserve_range=True)

				if self.options.computeMeanImage:
					img = img - self.meanImg

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

		imagesBatch = self.readImagesFromDisk([self.imageList[index] for index in self.indices], isTrain = True)
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

	def getTestBatch(self):
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
			imagesBatch = self.readImagesFromDisk([self.imageListTest[index] for index in self.indices], isTrain = False)
			labelsBatch = self.convertLabelsToOneHot([self.labelList[index] for index in self.indices])

		else:
			endIndex = self.currentIndexTest + self.options.batchSize
			if endIndex > self.totalImagesTest:
				endIndex = self.totalImagesTest
			self.indices = np.arange(self.currentIndexTest, endIndex)
			imagesBatch = self.readImagesFromDisk([self.imageListTest[index] for index in self.indices], isTrain = False)
			labelsBatch = self.convertLabelsToOneHot([self.labelListTest[index] for index in self.indices])
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

	def convertLabelsToOneHot(self, labels):
		oneHotLabels = []
		numElements = self.options.numClasses
		
		for label in labels:
			oneHotVector = np.zeros(numElements)
			oneHotVector[label] = 1
			oneHotLabels.append(oneHotVector)

		oneHotLabels = np.array(oneHotLabels)
		return oneHotLabels
