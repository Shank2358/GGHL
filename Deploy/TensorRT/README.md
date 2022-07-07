# GGHL-TensorRT in C++/Python
## Python
### Step1:Prepare the necessary package
- onnx
- onnxsim
- onnxruntime
- pycuda
- tensorrt
### Step2:Run the demo
You should get the tensorIR model(GGHL.onnx) by means of `torch2onnx.py`
You should change the absolute path including input_image,model weights,and tensorIR path,etc.
```python
main.py
```
## C++
### Step1:Prepare the necessary package
- TensorRT 8.4.2.15
- Opencv
- onnx2tensorrt
- cuda
- cudnn
### Step2: build the demo
You should modify the corresponding path in CMakeList.txt. If you use visual studio code as the basic editor. After installing the necessary plugin, please enter ctrel-shift-p to generate Make-file. In the next, generate the executable file by means of make.
Another choice:
```bash
    mkdir build
    cd build
    cmake ..
    make
```
Then run the demo. We will complement the input methods. You can modify the filename in the `main.cpp`. You should generate the tensorIR model which is exported by the `torch2onnx.py`
```
./GGHL
```
