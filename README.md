# CV_Sport_Project

A simple, yet effective way to analyze players and teams in various sports settings using computer vision and machine learning.

## Getting Started
Make sure you have [OpenCV 4](https://docs.opencv.org/4.x/d7/d9f/tutorial_linux_install.html) installed as well as CMake.

## Setting up the environment
To set up the environment, build the Singularity image using the following command:
```bash
cd OpenCV_Environment
sudo singularity build cv_sport_project.sif opencv_env.def
```

## Installing
Clone the repository and run the following commands:
```bash
mkdir build
cd build
cmake ..
cmake --build .
```

## Running
To run the program, run the following command from inside the build directory:
```bash
./main ../models/best.onnx <path to images folder> <path to output folder>
```

For the singularity version:
```bash
singularity exec ../OpenCV_Environment/cv_sport_project.sif -c "../models/best.onnx <path to images folder> <path to output folder>"
```

```bash
./model_inference <path_to_model> <path_to_image>
```

## Authors
Enrico D'Alberton - Player classification <br>
Federico Gelain - Playing Field detection and segmentation <br>
Pietro Girotto - Player segmentation <br>

## Other
Have a look at the [Road Map](./roadmap.md) to log and remember what is left to do.