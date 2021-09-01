# Welcome to the dataloaders directory!

### Description of scripts:

 - dataloader_Cityscapes.ipynb :

    - Reads in preprocessed Cityscapes images and labels from *"root/BiasMitigation/Datasets/Cityscapes"*
    
    - Creates two dataloader classes, **DatasetTrain** and **DatasetVal** which are used in the *trainer_merger.ipynb* and *main.ipynb* scripts.
    
    - **DatasetTrain**: 
        - Performs augmentation to the images consisting of random flipping, scaling and cropping to 256 x 256 read to be input into the network.
        - Extracts bias labels of the raw images. These are the integer colour values of each RGB channel binned into 8 classes. Created by dividing the 0-255 values by 32 and taking the floor value. This gives an array of values between 0-7 to perform categorisation. 
        - Label images are resized to the same as the raw images for semantic classification
        - Bias labels are resized to the same size of the feature map where the bias fork is located.
        - Raw images are normalized with pretrained imagenet mean and standard deviation values and outputted.
        
    - **DatasetVal**:
        - performs no augmentation
        - outputs label images, bias labels and raw images and image ID's to be fed into the network for validation. 
        
        
 - dataloader_SYNTHIA.ipynb : 
 
    - Reads in preprocessed SYNTHIA images and labels from *"root/BiasMitigation/Datasets/SYNTHIA"*
    - All other steps as above for Cityscapes. 

____