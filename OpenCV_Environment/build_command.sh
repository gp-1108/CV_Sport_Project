# Author: Pietro Girotto
# ID: 2088245
mkdir -p opencv_build 
cd opencv_build
cmake ..
cmake --build .
./model_inference ../models/best.onnx ../Images/im1.jpg