# Welcome to the helpers directory!

This directory contains useful code that came in handy throughout the project. The scripts themselves are not needed to run any models. 

A description of the scripts are as follows:

### *Tidier.ipynb*:

- Tidy files removing all but the last checkpoint file. Unless otherwise stated.

### *testing_image_manipulations.ipynb*:

- Code for checking the image manipulations are working correctly. 
- Create the bias label's and the human readable interpretation.
- Change the severity of the colour jitter to decide on parameters.
- Uses torch.transform library to add colour invert.

### *mock_dataloader.ipynb*:

- Replicates the dataloaders using the option flags. Checks to see if the dataloaders are inputting the data we expect them to. Included methods are added to the project utils and called into the *train_step*'s in the *trainer_merger.ipynb* script. 