## Facial-Emotion-Recognition
Facial emotion recognition on FER2013 dataset.
CSCE 633 project @ TAMU

# Understaing the project
In this project various machine learning models were used for facial emotion recognition on the FER2013 dataset. The data set has 32x32 blackend white images, each image representing one out of 6 emotions. Human accuracy on the dataset is around 65%. This is particularly a hard dataset to work with since it has a lot of misclassifications and random images that are not faces. The dataset is also highly imbalanced. As a result it is very hard to build models that correctly classify the classes with lower number of samples.

[Dataset Split](https://github.com/nitinchakravarthy/Facial-Emotion-Recognition/blob/master/images/fer2013.png)  

We started with a basic SVM, improved it with PCA. Then we moved to shallow CNNS (1,2 layers) like LeNet and AlexNet. Further we explored deep networks, VGG12, VGG16. 
Then we moved on to feature based methods. We extracted Histogram of oriented gradients and facial landmark features on the dataset and implemented SVM with PCA and conv nets on it.
As a last step of improvement we tried replacing the softmax layer of VGG12 with a SVM classifer layer at the end of the CNN. 

[Problem statment](https://github.com/nitinchakravarthy/Facial-Emotion-Recognition/blob/master/images/Picture1.png)  



We also looked investigated into visualizing the layer maps of each CNN in order to understand why each network classifer mis-classifies a few classes. More details about this are provided in the presentation and the report which are a part of the repo.

# Project structure
* ```Code```: Code for various models and training and feature extraction
* ```imgs```: Model training plots and feature example images and other images
* ```models ```: Some pretrained models that you can chill with

# Model Definitions
These are how the architechtures of the CNN models used. The hyperparmeters were tuned for the best average performance. So the actual model hyper parameters might not match the hyper parameters of the architechture.

[AlexNet(https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)](https://github.com/nitinchakravarthy/Facial-Emotion-Recognition/blob/master/images/5-layer.png)  

[LeNet(http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf)](https://github.com/nitinchakravarthy/Facial-Emotion-Recognition/blob/master/images/lenet.png)  

[VGG16(https://arxiv.org/abs/1409.1556)](https://github.com/nitinchakravarthy/Facial-Emotion-Recognition/blob/master/images/16-layer.png)  

# Features
We used 2 features. "Histogram of oriented gradients" and "Facial Landmark features"

[Histogram of orinted gradients and Facial Landmark features](https://github.com/nitinchakravarthy/Facial-Emotion-Recognition/blob/master/images/hog.png height="100")  

# Results
These are the confusion matrix and the auroc curves for the best model.
[](https://github.com/nitinchakravarthy/Facial-Emotion-Recognition/blob/master/images/cnn-confmat.png)  
[](https://github.com/nitinchakravarthy/Facial-Emotion-Recognition/blob/master/images/auroc.png)  


# Layer Visualization
These are the layer filters applied maps, of the VGG12 network in the first layer. For deeper layers look in the presentation or the report
[Layer 1](https://github.com/nitinchakravarthy/Facial-Emotion-Recognition/blob/master/images/vis1.png)  




