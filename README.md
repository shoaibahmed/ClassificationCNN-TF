# ClassificationCNN-TF

Implementation of complete CNN training pipeline in TensorFlow. The data reading mechanism is based on the TF Dataset API.
<br/>The trainer currently supports three architectures including Inception ResNet v2, ResNet-152 and NASNet all based on Slim implementations (https://github.com/tensorflow/models/tree/master/research/slim).

<br/>To train on a custom dataset, create a separate file for train, test and validation containing the image names and the corresponding label ids (from 0 to num_classes-1).

<br/>To initiate the training, use the command:
```
python trainer.py -t -s -v -c --batchSize 10 --trainingEpochs 10 -m NAS --displayStep 1 --weightedSoftmax
python trainer_v2.py -t -s -v -c --batchSize 10 --trainingEpochs 10 -m NAS --displayStep 1 --weightedSoftmax --trainSVM --l2Regularizer --reconstructionRegularizer
```
where -t stands for training, -s for training from scratch, -m defines the model to be used (IncResV2, ResNet, NAS), -c stands for model testing after training, and --weightedSoftmax uses weighted softmax with weights proportional to inverse frequency which helps in dealing with unbalanced classes. Training of SVM on the final feature vector is also possible by passing --trainSVM flag to trainer_v2.py along with --l2Regularizer which adds L2 regularization on the global pool feature vector and --reconstructionRegularizer which inserts a decoder network for reconstruction of the input to make sure the feature vector models all the main sources of variation in the input modality.

<br/>To initiate the testing phase, use the command:
```
python trainer.py -c -batchSize 10 -m IncResV2 --testDataFile ./data-labels-test.txt
```
where -c stands for testing and --testFileName defines the file name to be used for testing.

<br/><b>Note:</b> isTraining is always set to false for models due to tensorflow batch-norm issue.

<br/><br/> Author: <b>Shoaib Ahmed Siddiqui</b>
<br/> Email: <b>12bscsssiddiqui@seecs.edu.pk</b>
