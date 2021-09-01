# Welcome to the utils directory!

### Description of scripts:

 - *preprocess_cityscapes.ipynb* : 
     - Applys a mapping function to map the pixel values in the Cityscapes from 33 classes to 20 classes which we need for evaluation. 
     - It then runs through the label images and performs a similar operation creating ground truth labels for the new classes. Saves to disk in *'root/Dataset/Cityscapes/meta'* directory 
     - Computes class weights by keeping a running total of the number of pixels in each category classes and divides by the total number of pixels in all images in the training set. Outputting a vector of weights. This ensures loss functions are balanced by class. Saves pickle file disk. 

-  *preprocess_SYNTHIA.ipynb* : 
    -  The SYNTHIA dataset subset *synthia-rand-cityscapes* is provided with instance label rather than semantic labels and has 3 extra classes rather than the 20 classes needed for evaluation.
    -  The function loops over each image in the SYNTHIA dataset and performs a mapping function creating exactly the same classes as in the cityscapes dataset for a fair comparison. 
    -  The function then loops through each ground truth label mask, pulls out the first channel which has the semantic class information and computes a similar mapping function. 
    -  The class weights are computed as above and saved to disk.
    -  The dataset is split into a 70/30 training/validation set with 6500 training images and 2900 validation images. 

Rather than piling code into one utils script each is split into seperate scripts depending on where it is used in the wider code... 

- *utils_Deeplab.ipynb*
    - Contains a weight decay function used for both Deeplab and SegNet applied in the *trainer_merger.ipynb* script when initialising the optimiser, eithe ADAM or SGD.
    - Has mapping functions used in the preprocessing scripts which maps RGB ----> BGR within the Cityscapes classes to be used interchangably with CV2 and torch.


- *utils_LNTL.ipynb*
    - Contains the *save_option* method which saves the *option.json* file on model initialisation. 
    - Contains the *logger_setting* method which sets the format and content of the logger function outputting to the *train.log* file in the *training_logs* directory.
    - Contains the methods used to create the *test_input_images* The methods include: 
        - An interpretation of the bias labels. Here we multiply the 8 colour bins (0-7) by 32 and then add 16 to the total giving a human readable version of the bias label at the midpoint of each bin. Each channel R, G, B is output.
        - A post processing method whihc un-normalises the normalised input images for human veiwing. 
        - A detatch function to view and save the bias labels without affecting the network graph. 
    
- *utils_synthia.ipynb*
    - Contains the mapping funtions used by the synthia preprocessing script. 