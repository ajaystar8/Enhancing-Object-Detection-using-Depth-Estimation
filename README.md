# Exploring YOLOv3’s Potential: Object Detection with Depth Data

## Downloading the dataset

1. Download the `Labelled Dataset (~2.8GB)` from [NYU Depth Dataset V2](https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html). After the download is complete, you would have a `.mat` file. 
2. Place this file inside the `resources` folder. 
   3. Execute the `create_dataset.py` file, which is placed inside the `data` folder to get access to the images and depth maps.
3. To generate the HHA image dataset, run the `Depth2HHA/getHHA.py`. This will place the HHA images inside the `NYUD` directory. 

## Training the Model

1. Execute the `train.py` script by running the main method.
2. Enter the model you would like to train. We offer support to train `YOLOv3`, `ResYOLOv3 (additon concat)`
   and `ResYOLOv3 (depth concat)`. However, as explained in our report, `YOLOv3` is the best performing model.
3. We offer three training modes: `RGB`, `HHA` and `Fusion`. For training the `YOLOv3` model, you should use the `RGB`
   mode. For other, you can choose between `HHA` and `Fusion` modes. `HHA` mode trains the model just on HHA images
   and `Fusion` mode trains the model using both the RGB and HHA images.
4. We also support model logging and checkpoint saving, hence once the mode is chosen, follow the prompts to set up the
   filenames for the log file and checkpoint file.

## Making predictions using Model Checkpoints

_Note: Ensure that the checkpoint being loaded (you could change this by updating the path to the checkpoint) is for the
   model you are using to make a prediction. Otherwise, runtime exceptions will be thrown._

_Note: Due to restrictions in file size, we have placed only the model checkpoint for Experiment-1 (YOLOv3). In case you require the checkpoints for other models, please reach out to us. We will be happy to provide it to you._

1. Place the model checkpoints inside the `checkpoints` folder.
2. To make predictions, simply execute the `show_image.py` file.

```shell
python3 show_image.py
```

```ascii
>> Enter model type to train (yolov3 or resyolov3_add or resyolov3_depth): yolov3
Enter the checkpoint file name (include .pth.tar): Experiment_1.pth.tar
=> Loading checkpoint from checkpoints/Experiment_1.pth.tar
Checkpoint loaded successfully!
```
