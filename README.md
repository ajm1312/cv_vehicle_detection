# Vehicle Detection and Classifier

This is a repository for a vehicle detection and classifier, 
offering real-time detection and classification of 8 different
classes of vehicles.

These models were built on a standard YOLOv8 model. The documentation
for YOLO models can be found [here](https://docs.ultralytics.com/).

# Quickstart
A virtual environment is recommended for this project. This repository was tested using a conda environment, but any virtual environment should work.
You can find [documentation](https://docs.conda.io/en/latest/) for conda here. 

To install all required packages, follow these commands:
```bash
conda create --name cv python=3.10
conda activate cv
git clone https://github.com/ajm1312/cv_vehicle_detection.git
cd cv_vehicle_detection
pip install -e .
```

To train the model:
```bash
cd classifier
python3 train.py
```

To evaluate the model and test it on a video:
```bash
cd classifier
python3 evaluate.py
python3 test.py
```
The evaluation graphs will be saved to /runs/detects/evaluation.

All config settings are contained in the config.yaml file. To change the tested model or the video to test the model on, change the path in
that file.

# Packages
The packages used in this project are:
* numpy
* matplotlib
* kagglehub
* ultralytics
* opencv-python
* pandas