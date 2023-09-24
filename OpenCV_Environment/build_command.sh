# Author: Pietro Girotto
# ID: 2088245
mkdir -p opencv_build 
cd opencv_build
cmake ..
cmake --build .
./main ../models/best.onnx ../Sport_scene_dataset ../Model_output