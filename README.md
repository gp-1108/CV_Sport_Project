# CV_Sport_Project

A simple, yet effective way to analyze players and teams in various sports settings using computer vision and machine learning.

## Getting Started
Make sure you have [OpenCV 4](https://docs.opencv.org/4.x/d7/d9f/tutorial_linux_install.html) installed as well as CMake.

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
./model_inference <path_to_model> <path_to_image>
```

## Authors
Enrico D'Alberton - Player classification <br>
Federico Gelain - Playing Field detection <br>
Pietro Girotto - Player segmentation <br>

## Other
Have a look at the [Road Map](./roadmap.md) to log and remember what is left to do.