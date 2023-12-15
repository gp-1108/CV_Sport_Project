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

For the singularity version:
```bash
singularity exec ../OpenCV_Environment/cv_sport_project.sif bash ../OpenCV_Environment/build_command.sh
```

NOTE: The singularity command both builds and runs the program.

## Running
To run the program, run the following command from inside the build directory:
```bash
./main ../models/best.onnx <path to images folder> <path to output folder>
```

To modify the singularity runtime arguments, edit the build_command.sh file.


## Authors
Enrico D'Alberton (https://github.com/enricopro) - Player classification <br>
Federico Gelain (https://github.com/FedericoGelain) - Playing Field detection and segmentation <br>
Pietro Girotto (https://github.com/gp-1108) - Player segmentation <br>
