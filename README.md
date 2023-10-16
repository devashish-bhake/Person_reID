# Dataset collection and preprocessing
## Person Detection
### Dataset collection
For solving this problem, CrowdHuman Open-Source Dataset would be the most apt dataset in the yolo format. The dataset consists of 4355 images of annotated people in the wild. The dataset is divided into training, validation and testing sets. The training set consists of 3485 images, the validation set consists of 870 images.
### Preprocessing
The preprocessing of the images involves applying an auto-orient to the images and sizing them to 416x416.
### Training 
The model has been trained on the above dataset and will be used in the further processes
### Person Identification and Tracking
The person identification and tracking is implemented on the video of the crowd. The video is first converted into frames and then the frames are passed through the model to detect the people in the frame. And as far as tracking through multiple video inputs or cameras is concerned it has been implemented using multithreading wherein multiple threads are running the same detection function parallely so that any change in one will be detected by the other and the tracking will be done accordingly. 
## Person Re-identification
### Dataset collection
The collection of dataset involves downloading the Market-1501 dataset which consists of 32,668 annotated bounding boxes of 1,501 identities. The dataset is divided into training and testing sets. The training set contains 12,936 images of 751 identities, while the testing set contains 19,732 images of 750 identities. The dataset is collected from public street view cameras with different cameras as well. The image annotations show which camera id has the image been captured from.
### Preprocessing
The preprocessing of the dataset involves organising the data in a particular folder format in which the images with the same 'id' are kept in the same folder which the format expected by the torch vision library.
### Training
In order to begin the training first we have to organise the dataset in the format it is required by the torchvision library
1. The dataset is organised in the following format:
```
├── Market/
│   ├── bounding_box_test/          /* Files for testing 
│   ├── bounding_box_train/         /* Files for training 
│   ├── gt_bbox/                    /* Files for multiple query testing 
│   ├── gt_query/                   /* We do not use it 
│   ├── query/                      /* Files for testing (query images)
│   ├── readme.txt
```
2. The next step is to convert the dataset in the following format:
```
├── Market/
│   ├── bounding_box_test/          /* Files for testing 
│   ├── bounding_box_train/         /* Files for training 
│   ├── gt_bbox/                    /* Files for multiple query testing 
│   ├── gt_query/                   /* We do not use it
│   ├── query/                      /* Files for testing (query images)
│   ├── readme.txt
│   ├── pytorch/
│       ├── train/                   /* train 
│           ├── 0002
|           ├── 0007
|           ...
│       ├── val/                     /* val
│       ├── train_all/               /* train+val      
│       ├── query/                   /* query files  
│       ├── gallery/                 /* gallery files  
│       ├── multi-query/    
```
In the above format the train folder contains the folders of the ids and the images of the particular id are kept in the folder of that id. The same is done for the validation set as well. The query folder contains the images of the query set and the gallery folder contains the images of the gallery set. The multi-query folder contains the images of the multiple query set.

3. The next step is to define the model that we want to use in this case I have decided to go with the HRNet (High-Resolution Net) ImageNet model for the task. The model is defined in the ```model.py``` file.

4. The next step is to define the loss function that we want to use in this case I have decided to go with the Circle Loss function. The loss function is defined in the ```circle_loss.py``` file. 

5. The final step is to train the model using the ```train.py``` file. The model is trained for 60 epochs and the model is saved after every 10 epochs. The model is trained on the GPU and the training time for 60 epochs is around 2 hours.

In order to run the training code the following command is used:
```
python train.py --gpu_ids 0 --name reid_hrnet --use_hr True --train_all --batchsize 32  --data_dir directory_of_training_data
```
The final results of the training of the model is:

![result](https://github.com/devashish-bhake/Person_reID/blob/main/personReID/model/reid_hrnet/train.jpg?raw=true)

### Testing
The testing of the model is done using the ```test.py``` file. The model is tested on the query set and the gallery set. The model is tested on the GPU and the testing time is around 10 minutes. The model is tested on the query set and the gallery set and the results are saved in the ```result.txt``` file. The results of this code is a matlab file which contains all the extracted features of all the query images.
In order to run the testing code and generate the characteristic descriptions of the query images the following command is used:
```
python test.py --gpu_ids 0 --name ./model/reid_hrnet --test_dir path_to_testing_set  --batchsize 32 --which_epoch 60 
```
the --which_epoch parameter will load the checkpoint that you want to test since I trained the model for 60 epochs I loaded the 60th epoch checkpoint which is the final model.

### Evaluation
The evaluation of the model is done using the ```evaluate_gpu.py``` file. The evaluation is done on the GPU and the evaluation time is around 10 minutes. Th evaluation results are as follows: 
```
Rank@1:0.909145
Rank@5:0.965558
Rank@10:0.977138
mAP:0.764341
```

### Demo Outputs
The demo output has been generated using the query images of the dataset and the gallery images of the dataset. The demo output is generated using the ```demo.py``` file. The demo output is generated on the GPU.
The demo outputs look like this:

![output](https://github.com/devashish-bhake/Person_reID/blob/main/personReID/show.png?raw=true)

In the above outputs we can see that the model was able to identify various images that were of the similar in the characteristic description of the query image.
