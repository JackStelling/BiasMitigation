# Welcome to the evaluation directory!

This directory contains the evaluation scripts used for analysing results from the training logs.

A description of the scripts are as follows:

### *eval_for_val_for_metrics.ipynb*:

- Creates label images from the predicted output of the network for the best trained model. These label images are then saved to disk and are used in the official Cityscapes metric script to get a detailed report of mIoU metrics. 

### *eval_on_val*:

- Generates qualatative results from loading in best trained models during training experiments and produces an overlaid image of the prediction mask and the raw image and saves the resultant images to disk.

### *visualisations.ipynb*:

- Used to generate adhoc analysis of the losses from reading in the pickle files saved to disk during training cycles.
- Useful to visually compare the various experiments. 
