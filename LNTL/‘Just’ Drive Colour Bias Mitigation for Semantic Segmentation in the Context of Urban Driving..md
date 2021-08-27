# **'Just' Drive** : Colour Bias Mitigation for Semantic Segmentation in the Context of Urban Driving.

## *Contents:*
**1.** &nbsp; &nbsp; Introduction
**2.** &nbsp; &nbsp; Requirements and environment
**3.** &nbsp; &nbsp; Datasets and downloads
**4.** &nbsp; &nbsp; Cited Repositories
**5.** &nbsp; &nbsp; How to run this Repository
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; -- *Setting up the file structure*
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; -- *Preprocessing Data*
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; -- *Training*
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; -- *Evaluation*
**6.** &nbsp; &nbsp; Licensing
**7.** &nbsp; &nbsp; Summary 

</br>

## 1. Introduction:

Welcome to the repository on bias mitigation in the context of semantic segmentation. This repository is the pytorch implementation of the project found here:

'link to final pdf'

The project aim is to push deep convolutional neural networks to 'unlearn' colour information whilst performing a pixel-wsie semantic segmentation task. Autonomous vechicles have been selected as a use case, as a safety critical AI technology - which is accountable for high stakes decisions constantly - meaning just and rigour is essential.

The project builds on top of other work in the field which have shown that biases exist within the CNNs slection of which cues to use. If colour information is available the model will surely use it as a cue for making decisions - even if it is not the correct cue to use. For example in the case of road driving scenes: Is a tree still classified as a tree even if it doesnt have its summer crown of leaves? Or is a road still classified as such even though a winter's snow has left a white blanket over the city? If we are to depend on AV technology as the transportation method of the future we must strive for a generalisable and equitable system. 

The aim of the project was to remove these biases which 



Biases exist in all aspects of life, if the bias is known it can be accounted for - if the bias is unknown it acts as a much more insidious force. Further work which will be conducted after this project will focus on the removal of an unknown bias by use of latent representation using a VAE, more information regarding future work can be found in the appendix of the main report. 

Altough this project focusses on bias removal in the context of semantc segmentation for autonomous vehicles the concept itself is ubiqutous and can be applied to any situation where colour bias is suspected to exist within the input data. 

</br>

## 2. Requirements and Environment

The project was completed using Google Colab with ...specs

**Package Versions**
- Python : **3.7.1**
- Pytorch : **1.9**
- CUDA : **11.0221**
- OS : **Linux 64 bit** 
- Google Colab : 
    - CPU: **150GB**
    - GPU: **12GB**

Output of nvidia-smi in the Colab environment
```console
user: $ nvidia-smi 

> nvcc: NVIDIA (R) Cuda compiler driver
> Copyright (c) 2005-2020 NVIDIA Corporation
> Built on Wed_Jul_22_19:09:09_PDT_2020
> Cuda compilation tools, release 11.0, V11.0.221
> Build cuda_11.0_bu.TC445_37.28845127_0
> Tue Aug 17 09:19:30 2021

+-----------------------------------------------------------------------------+
| NVIDIA-SMI 470.57.02    Driver Version: 460.32.03    CUDA Version: 11.2     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  Tesla P100-PCIE...  Off  | 00000000:00:04.0 Off |                    0 |
| N/A   46C    P0    28W / 250W |      0MiB / 16280MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+
```

Features of Colab you may need to edit if you want to run locally or elsewhere:
- Most scripts start with a mount drive command to access files on Google Drive. 
- File paths will need to be ammended depending on where you run the script. 
- File imports using *.ipynb* require slightly different syntax so most import statements have the module *import_ipynb* to handle this standard *.py* scripts don't require this.
- The command 'cv2.imshow' from Pythons computer visions package doesnt work in Colab so we run a patch allowing 'cv2_imshow' instead - outside of Colab this will need to be ammended.
- A useful command line tool to convert the scripts from *.ipynb* -----> *.py*  can be found here: https://nbconvert.readthedocs.io/en/latest/ 

</br>

## 3. Datasets and Downloads:

This project uses two main datasets: **Cityscapes** and **SYNTHIA**. 

#### **Cityscapes** 
The Cityscapes dataset is a widely used semantic segmentation benchmark. Cityscapes contains data from 50 cities in Europe. The datset contains over 5000 fine annoted images at a resolution of 2048 x 1024 providing pixel-level, instance-level and panoptic semantic ground truth labels. 

- Visit the official cityscape website at: https://www.cityscapes-dataset.com/ for an in depth description of the dataset.
- Provide your email address and some contact information to gain access to the dataset downloads portal.
- Once access is granted download the *gt-fine-trainvaltest.zip*  as shown below:

[]insert cityscapes download image here.(might be overkill)

#### SYNTHIA
The Synthia dataset is comprised of photo-realistic images rendered from a virtual city. The dataset used in this paper is the *synthia-rand-cityscapes* subset containing 9400 images at a resolution of 1280 x 760; which contain labels compatible with Cityscapes, allowing for fairer cross examination of results.

- Visit the official Synthia website at http://synthia-dataset.net/
- Navigate to the downloads tab and download the *synthia-rand-cityscapes* subset for analysis. 
- An extra validation set is used in the project using a winter 


</br>

## 4. Cited Repositories:

This project builds on top of some other *Git* repositories, I thank the authors and use the code respectfully and transparently:

- For DeeplabV3 implementation and some Cityscapes helpers:
    - *https://github.com/fregu856/deeplabv3*
- For SegNet implementation:
    - *https://github.com/say4n/pytorch-segnet*
- For Cityscapes evaluation scripts:
    - *https://github.com/mcordts/cityscapesScripts*
- For Learning not to Learn scheme:
    - *https://github.com/feidfoe/learning-not-to-learn*

</br>

## 5. How to run this repository

Enter the space you wish to clone this *Git* repository to, I highly recommend this to be Google Colab. 

Hereafter the steps needed to run in Colab are discussed. Each directory also has its own README which dicusses exactly what the code within aims to do. 


### Setting up file structure

- From a terminal navigate to the repository destination and enter:
```console
user: $ sudo dnf install git-all
user: $ git version
user: $ git clone https://github.com/JackStelling/BiasMitigation
```

- Upload the datasets into the correct locations from the links provided in Section 3. They must be added using the following file structure for the paths in the scripts to work. 

![filestructure](https://i.imgur.com/UPzKgkH.png)



- Navigate inside the *'BiasMitigation'* project repository and clone the official gcityscapesScripts GitHub repository: 

```console
user: $ cd /content/drive/MyDrive/BiasMitigation
user: $ git clone https://github.com/mcordts/cityscapesScripts
```
 - Enter the cloned cityscapesScripts repository and delete their .git log if you wish to track the BiasMitigation using Git. 

### Preprocessing Data

- Navigate to 
```console
user: $ /content/drive/MyDrive/BiasMitigation/LNTL/utils
```
- Run:
    - *preprocess_cityscapes.ipynb*
    - *preprocess_SYNTHIA.ipynb*
    
    these files **ONLY NEED TO BE RAN ONCE** they add all preprocessing neccessary for both cityscapes and SYNTHIA more information can be found in the 'utils' folder README. These scripts take approx 6hours heach to run on the environemnt specified above. 

### Training

- Open *main.ipynb* the file contains many flags to set up the different experiments mentioned in the main project report. Examples: 
    - To train a baseline Deeplab model without the Learning Not To Learn Scheme with Cityscapes colour training images use the following options whilst keeping all other at default:
        ```python
        exp_name = Exp1_Stage1 # <--- We MUST change this at each experiment to avoid saving over files
        train_baseline = True
        train_greyscale = False 
        network_type = 'Deeplab'
        dataset = 'Cityscapes'
        ```
    - To train a SegNet network with the Learning Not To Learn scheme with SYNTHIA training images and colour jittered validation images use the following options whilst keeping all other at default: 
       ```python
        exp_name = Exp2_Stage1
        train_baseline = False
        train_greyscale = False
        val_only_jitter = True 
        network_type = 'SegNet'
        dataset = 'SYNTHIA'
        ```
    - To train a Deeplab network with the Learning Not To Learn scheme using the Cityscapes dataset using a pretrained model use:
        ```python
        exp_name = Exp3_Stage1
        train_baseline = False
        checkpoint = root + 'LNTL/training_logs/Deeplab/Cityscapes/LNTL/Exp5_Stage1/checkpoints/checkpoint_bias_head_epoch_0039.pth
        network_type = 'Deeplab'
        dataset = 'Cityscapes'
        ```
- Other functionality exists within the code, explore the options list to find out more. All code modifications only need to be made inside the *'main.ipynb'* script.

- Model Timings with quoted environment:
    - Each model takes approx. **12 hours** to train for 100 epochs on the Cityscapes datset. SegNet is slightly quicker as is the baseline model. 
    - Each model takes approx. **16 hours** to train for 100 epochs on the SYNTHIA datset. Again SegNet is slightly quicker as is the baseline model. 


- All model runs will create training_logs and diagnostic graphs in the correct locations,see the README in the *'training_logs'* directory for the file tree diagrams.  

- Test images are drawn from the train and validation dataloaders and saved to disk in the '~root/training_logs/{model selection options}/test_input_images' to ensure that the correct training data is being loaded into the model. 


### Evaluation

Loss curves can be inspected during model training. For a more granular analysis a few techniques used:

1. **Via visualisation :** analysing the loss pickles
    - Step by step instructions:
        - Navigate to project evaluation scripts
        ``` console
        user: $ cd /content/drive/MyDrive/BiasMitigation/LNTL/evaluation
        ```
        - Open the *visualisations.ipynb* file
        - Run all cells and loss curves of all experiments are created
        - This is designed to be a self evaluation tool and best used editing the script yourself and analysing output. 

2. **Qualitatively :** Preating overlaid images from the predictions and the raw images files.
    - Step by Step instructions:
        - Navigate to project evaluation scripts
        ``` console
        user: $ cd /content/drive/MyDrive/BiasMitigation/LNTL/evaluation
        ```
        
        
3. **Quantatively :** Passing the best model through the citycapesScripts evaluation server. 
    - Step by step instructions:
        -  Navigate to the shell script for evaluation
        ``` console
        user: $ cd /content/drive/MyDrive/BiasMitigation/LNTL/evaluation/shell_script_for_metrics
        ```

</br>

## 6. Licensing

**BiasMitigation License:**

Copyright (c) 2021 Jack Stelling

**SegNet License:**

Copyright (c) 2018 Sayan Goswami

**Deeplab License:**

Copyright (c) 2018 Fredrik Gustafsson

**License Statement for all associated repositories:**

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

</br>

## 7. Summary 

Give a brief project conclusion and some areas of future work. 

### Contact Information

If you have any questions/concerns with the code provide or would like to collaborate on future work within the sphere of bias mitigation in deep learning please get in touch:

**j.stelling2@ncl.ac.uk** 

</br>