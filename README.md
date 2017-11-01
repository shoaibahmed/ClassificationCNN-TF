# ClassificationCNN-TF

Implementation of complete CNN training pipeline in TensorFlow. The data reading mechanism is based on the TF Dataset API.
<br/>The trainer currently supports three architectures including Inception ResNet v2, ResNet-152 and NASNet all based on Slim implementations (https://github.com/tensorflow/models/tree/master/research/slim).

<br/>To train on a custom dataset, create a separate file for train, test and validation containing the image names and the corresponding label ids (from 0 to num_classes-1).

<br/>To initiate the training, use the command:
```
python trainer.py -t -s -v --batchSize 10 --trainingEpochs 10 -m NAS
```
where -t stands for training, -s for training from scratch and -m defines the model to be used (IncResV2, ResNet, NAS).

<br/>To initiate the testing phase, use the command:
```
python trainer.py -c -batchSize 10 -m NAS
```
where -c stands for testing.

<br/><br/> Author: <b>Shoaib Ahmed Siddiqui</b>
<br/> Email: <b>12bscsssiddiqui@seecs.edu.pk</b>
