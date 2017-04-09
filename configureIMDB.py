import os
import random
import shutil
from os import listdir
from os.path import isfile, join
from optparse import OptionParser

def traverseDirectory(options):
	imagesFileTrain = open(options.imagesTrainOutputFile, 'w')
	imagesFileTest = open(options.imagesTestOutputFile, 'w')
	classes = {"edeka": 0, "jysk": 1, "meinladen": 2, "netto": 3, "post": 4, "saturn": 5, "other": 6}

	# Extract class name
	if os.name == 'nt':
		separator = '\\' # Windows
	else:
		separator = '/' # Ubuntu

	keysCounter = 0
	print ('Processing training set images')
	for root, dirs, files in os.walk(options.trainDir):
		path = root.split('/')
		print ("Directory:", os.path.basename(root))
		
		for file in files:
			if file.endswith(options.searchString):
				fileName = str(os.path.abspath(os.path.join(root, file)))
				fileNameList = fileName.split(separator) 

				# Class Name and Number
				className = fileNameList[-2]
				if className in classes:
					pass
				else:
					classes[className] = keysCounter
					keysCounter += 1
				classNumber = classes[className]
				
				imagesFileTrain.write(fileName + ', ' + str(classNumber) + '\n')

	print ('Processing test set images')
	for root, dirs, files in os.walk(options.testDir):
		path = root.split('/')
		print ("Directory:", os.path.basename(root))
		
		for file in files:
			if file.endswith(options.searchString):
				fileName = str(os.path.abspath(os.path.join(root, file)))
				fileNameList = fileName.split(separator) 

				# Class Name and Number
				className = fileNameList[-2]
				if className in classes:
					pass
				else:
					classes[className] = keysCounter
					keysCounter += 1
				classNumber = classes[className]
				
				imagesFileTest.write(fileName + ', ' + str(classNumber) + '\n')


	# Write classes name to file
	classesFile = open(options.classesOutputFile, 'w')
	classes = [(k, classes[k]) for k in sorted(classes, key=classes.get)] # Convert dict to sorted list of tuples
	for key, value in classes:
		classesFile.write(str(value) + ' ' + key + '\n')
	classesFile.close()

	imagesFileTrain.close()
	imagesFileTest.close()

if __name__ == "__main__":

	# Command line options
	parser = OptionParser()
	parser.add_option("--trainDir", action="store", type="string", dest="trainDir", default=u".", help="Root directory of training data to be searched")
	parser.add_option("--testDir", action="store", type="string", dest="testDir", default=u".", help="Root directory of test data to be searched")
	parser.add_option("--searchString", action="store", type="string", dest="searchString", default=".jpg", help="Criteria for finding relevant files")
	parser.add_option("--imagesTrainOutputFile", action="store", type="string", dest="imagesTrainOutputFile", default="train.idl", help="Name of train images file")
	parser.add_option("--imagesTestOutputFile", action="store", type="string", dest="imagesTestOutputFile", default="test.idl", help="Name of test images file")
	parser.add_option("--classesOutputFile", action="store", type="string", dest="classesOutputFile", default="classes.idl", help="Name of file storing class names and associated labels")

	# Parse command line options
	(options, args) = parser.parse_args()

	traverseDirectory(options)

	print ("Done")
