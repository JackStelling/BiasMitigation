# Welcome to the models directory!

This directory contains the network classes for each of the networks used in the project. 

A description of the scripts are as follows:

### *segnet.ipynb*:

- Contains the offical implementation of SegNet from ... the code in this script is taken from ... with some ammendments to add in the bias removal form in the last layer of SegNet. The network loads in pretrain VGG network weights from ImageNet 

### *deeplab.ipynb*:

- Ties together the resnet and atrous spatial pooling pyramid modules which make up the DeepLabV3 network. 

### *aspp.ipynb*

- Creates the Atrous Spatial Pooling Pyramid module which is a crucial feature of the DeeplabV2, V3 and V3+ networks to understand objects of different scales.
- 
- The bias removal fork is added into the penultimate layer in the network immediately after concatenating the ASPP feature maps and is added to the output. 

### *resnet.ipynb*

- Creates the ResNet backbone of the DeepLab network. The network is initialised with pretrained ResNet weights from ImageNet.

### *biashead.ipynb*

- Creates the bias removal network class (denoted network *h* in the report and diagrams). This network is a fully convolutional network preserving spatial dimensions from its input feature map and performs a classification task using the *bias_labels* generated in the dataloaders as ground truth labels. 

___