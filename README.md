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
### Sample Output
Below is the sample output from the Person Identification model: 
![output_gif_1](https://github.com/devashish-bhake/Person_reID/blob/main/person_ID/output_video.gif)
![output_gif_2](https://github.com/devashish-bhake/Person_reID/blob/main/person_ID/output_video_2.gif)
If you want to run multiple video based inputs in the person identification script then it can easily be done by using the following instructions:
1. First locate the video_1, video_2...lines in ```personID.py``` file.
2. If you want to give a video input then just give it the path for the video.
3. Whereas if you want to give it a camera input that is connected via usb, then replace the video paths with camera ids like 0, 1, and so on
for example if you have 5 cameras, make video_1, video_2 and so on till video_5 and assign them int values from 0 to 4 respectively.
4. Now after that create multiprocessing threads at the end using the same format and start those threads to access all the cameras at the same time and the model as well
5. but beware that the more cameras you add the more load it will put on the system and the more VRAM and RAM will be needed to keep them running
A sample output of 2 videos running at the same time is attached below (I have run 2 videos only because my laptop has enough vram for running only 2 inference engines at a time but the code has been written to accomodate any number of inference engines as long as the necessary hardware is provided):

https://github.com/devashish-bhake/Person_reID/assets/79623853/86efe5f5-2849-4977-a3da-d9a6bfe557ef

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

## Licence Plate Recognition
### Dataset collection
The dataset collection involves downloading a dataset that contains about 1295 images of car number plates and the annotations of the number plates. The dataset is divided into training and testing sets. The training set contains 1100 images, the validation set consists of 144 images and the testing set contains 51 images. The dataset is collected from public street view cameras with different cameras as well. 

### Preprocessing
the dataset has been preprocess by applying an auto-orient filter and the images have also been resized to 640x640 for better training. Augmentations have also been applied to counter the low number of training examples that we have. The augmentations that have been applied are:
```
outputs_per_training_example = 2
grayscale = 0.25
blur = 2.5
noise = 0.05
```

### Training
the model has been trained on YOLOv8 using their python interface, and it has been trained upto 20 epochs. The model that was selected for solving this particular task was the ```yolov8s.pt``` since that model was the perfect balance between model complexity and model inference efficiency. The model was trained on the GPU and the training time for 20 epochs was around 1.5 hrs.

### Testing
The model was tested on 51 testing images that I created during the dataset creation and collection phase.

### Demo outputs
![demo](https://github.com/devashish-bhake/Person_reID/blob/main/LicencePlateID/output.JPG?raw=true)
The detected number plates have been stored in the sqlite database for further analysis and processing. The schema of the database looks like this:

![Screenshot from 2023-10-16 21-20-31](https://github.com/devashish-bhake/Person_reID/assets/79623853/9569984e-0bfe-41ec-85ae-111fa45dc92a)

