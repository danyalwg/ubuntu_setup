# **Step-by-Step Ubuntu Setup for Machine Learning with CUDA, TensorFlow, PyTorch, and TensorRT**

This guide walks you through the installation of essential tools and libraries for machine learning and deep learning development on Ubuntu, including NVIDIA CUDA, TensorFlow, PyTorch, TensorRT, and ONNX.

## **1. System Update and Upgrade**
Before starting, ensure that your system is up-to-date:

``` 
sudo apt update && sudo apt upgrade -y
```

### **Expected Result:**
The system repositories will be updated, and any outdated packages will be upgraded.

---

## **2. Install NVIDIA Drivers and CUDA Toolkit**
Ensure you have the NVIDIA drivers and CUDA toolkit installed.

### **2.1. Install the CUDA Toolkit:**

``` 
sudo apt install nvidia-cuda-toolkit
```

### **2.2. Verify CUDA and NVIDIA Driver Installation:**

Check the NVIDIA GPU status and CUDA version:

``` 
nvidia-smi
nvcc --version
```

### **Expected Result:**
- `nvidia-smi` should show details about your GPU, including the driver version.
- `nvcc --version` should return the version of the installed CUDA toolkit.

---

## **3. Install Python and Essential Packages**
Make sure Python 3 and `pip` are installed for package management.

### **3.1. Install Python `pip`:**

``` 
sudo apt install python3-pip
```

### **3.2. Configure pip:**

``` 
sudo bash -c 'echo -e "\n[global]\nno-cache-dir = false\nbreak-system-packages = true" >> /etc/pip.conf'
```

### **3.3. Install Python Packages for Error Handling:**

``` 
python3 -m pip install pretty_errors
python3 -m pretty_errors
```

### **Expected Result:**
- `pip` should be installed without any errors.
- `pretty_errors` should be installed to format Python errors more readably.

---

## **4. Install Machine Learning Libraries**
Install commonly used Python libraries for machine learning and data processing.

### **4.1. Install OpenCV:**

``` 
sudo apt-get install python3-opencv
```

### **4.2. Install Scientific Libraries:**

``` 
pip3 install --upgrade numpy scipy pandas matplotlib seaborn opencv-python-headless scikit-image pillow
```

### **4.3. Install Keras, PyTorch, and TorchVision:**

``` 
pip3 install --upgrade keras torch torchvision
```

### **4.4. Install Theano:**

``` 
sudo pip3 install theano
```

### **4.5. Install TensorFlow (with CUDA support):**

``` 
python3 -m pip install 'tensorflow[and-cuda]'
```

---

## **5. Install cuDNN and TensorFlow Libraries**
Install NVIDIA's cuDNN for optimized deep learning performance and ensure TensorFlow's GPU support.

### **5.1. Install cuDNN:**

``` 
sudo apt-get install libcudnn8 libcudnn8-dev
```

### **5.2. Install TensorBoard:**

``` 
pip3 install tensorboard
```

### **5.3. Install Progress Bar Library:**

``` 
pip3 install tqdm
```

### **Expected Result:**
- cuDNN will be installed to accelerate neural network training.
- TensorFlow-specific tools like `tensorboard` and `tqdm` will be available for use.

---

## **6. Install TensorRT**
TensorRT is essential for running high-performance inference on NVIDIA GPUs.

### **6.1. Download and Install TensorRT:**

``` 
sudo wget https://developer.download.nvidia.com/compute/machine-learning/tensorrt/8.6.1.6/ubuntu2004/x86_64/tensorrt-repo-ubuntu2004-8.6.1.6-ga-cuda11.7_1-1_amd64.deb
sudo dpkg -i tensorrt-repo-ubuntu2004-8.6.1.6-ga-cuda11.7_1-1_amd64.deb
sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/machine-learning/tensorrt/8.6.1.6/ubuntu2004/x86_64/7fa2af80.pub
sudo apt update
```

### **6.2. Install TensorRT Libraries:**

``` 
sudo apt install libnvinfer8 libnvinfer-dev libnvinfer-plugin8
```

---

## **7. Install Python API for TensorRT**
Install the TensorRT Python bindings to enable TensorRT in Python.

### **7.1. Install NVIDIA PyIndex:**

``` 
pip3 install nvidia-pyindex
```

### **7.2. Install TensorRT Python Bindings:**

``` 
pip3 install tensorrt
```

### **7.3. Verify TensorRT Installation:**

``` 
python3 -c "import tensorrt as trt; print(trt.__version__)"
```

### **Expected Result:**  
The installed version of TensorRT should be printed without errors.

---

## **8. Install TensorFlow-TensorRT and PyTorch-TensorRT**
Integrate TensorFlow and PyTorch with TensorRT for optimized inference.

### **8.1. Install TensorFlow-TensorRT:**

``` 
pip3 install tensorflow-tensorrt
```

### **8.2. Install PyTorch-TensorRT:**

``` 
pip3 install torch-tensorrt
```

---

## **9. Install Nsight Tools**
Nsight tools are used for profiling and debugging GPU applications.

### **9.1. Install Nsight Systems:**

``` 
sudo apt-get install nsight-systems
```

### **9.2. Install Nsight Compute:**

``` 
sudo apt-get install nsight-compute
```

---

## **10. Install ONNX for Model Interoperability**
ONNX enables the conversion of models between different deep learning frameworks.

### **10.1. Install ONNX:**

``` 
pip3 install onnx onnx-tf
```

### **10.2. Install PyTorch ONNX Support:**

``` 
pip3 install onnx-torch
```

---

## **11. Verifications**
After completing the setup, run the following verifications to ensure everything is working correctly.

### **11.1. Verify NVIDIA GPU and CUDA Installation:**

#### **Check NVIDIA GPU Status:**

``` 
nvidia-smi
```

**Expected Result:**  
- You should see a table with the following details:  
    - GPU model (e.g., "Tesla V100", "RTX 3090")
    - Driver version (e.g., "460.39")
    - CUDA version (e.g., "CUDA Version 11.2")
    - Memory usage details (used, free, total)

#### **Verify CUDA Installation:**

``` 
nvcc --version
```

**Expected Result:**  
- The command should output the version of the installed CUDA toolkit, for example:

```  
nvcc: NVIDIA (R) Cuda compiler driver Copyright (c) 2005-2021 NVIDIA Corporation Built on Sun_Apr_11_00:42:13_PDT_2021 Cuda compilation tools, release 11.2, V11.2.67
```


#### **Test CUDA with TensorFlow:**

``` 
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

**Expected Result:**  
- If TensorFlow detects a compatible GPU, it will return a list of available GPU devices, for example: 

``` 
[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```


- If no GPU is detected, the list will be empty:  

```
[]
```


---

### **11.2. Verify TensorFlow Installation (with CUDA support):**

#### **Check TensorFlow Version:**

``` 
python3 -c "import tensorflow as tf; print(tf.__version__)"
```

**Expected Result:**  
- This will print the installed TensorFlow version (e.g., `2.9.0` or another version compatible with CUDA).  
- Ensure that the installed version is compatible with the CUDA version (check the [TensorFlow compatibility guide](https://www.tensorflow.org/install/source#gpu) if unsure).

#### **Check TensorFlow GPU Support:**

``` 
python3 -c "import tensorflow as tf; print(tf.test.is_gpu_available())"
```

**Expected Result:**  
- If TensorFlow is correctly configured with GPU support, it should return `True`.  
- If it returns `False`, there might be an issue with the CUDA or cuDNN setup.

---

### **11.3. Verify PyTorch Installation (with CUDA support):**

#### **Check PyTorch Version:**

``` 
python3 -c "import torch; print(torch.__version__)"
```

**Expected Result:**  
- This will print the installed PyTorch version (e.g., `1.12.0`).  
- Ensure that the PyTorch version installed is compatible with your CUDA version (check the [PyTorch installation guide](https://pytorch.org/get-started/locally/) for compatibility).

#### **Verify GPU Availability in PyTorch:**

``` 
python3 -c "import torch; print(torch.cuda.is_available())"
```

**Expected Result:**  
- If PyTorch has been correctly configured with GPU support, this should return `True`.  
- If it returns `False`, the issue might be related to the CUDA or PyTorch installation.

---

### **11.4. Verify TensorRT Installation:**

#### **Check TensorRT Version:**

``` 
python3 -c "import tensorrt as trt; print(trt.__version__)"
```

**Expected Result:**  
- This will print the installed TensorRT version (e.g., `8.6.1`).  
- If the command fails, it indicates a problem with the TensorRT installation.

#### **Verify TensorRT Integration with TensorFlow:**

``` 
python3 -c "import tensorflow as tf; from tensorflow.python.compiler.tensorrt import trt_convert; print(trt_convert)"
```

**Expected Result:**  
- If TensorFlow-TensorRT integration is correctly set up, this will print the `trt_convert` module’s details without errors.
- If there's an error, check the version compatibility between TensorFlow and TensorRT.

---

### **11.5. Verify ONNX Installation:**

#### **Check ONNX Version:**

``` 
python3 -c "import onnx; print(onnx.__version__)"
```

**Expected Result:**  
- This should print the version of the ONNX library, for example, `1.11.0`.

#### **Check ONNX-TensorFlow Integration:**

``` 
python3 -c "import onnx_tf; print(onnx_tf.__version__)"
```

**Expected Result:**  
- This will print the installed `onnx-tf` version, confirming that ONNX models can be converted to TensorFlow format.

---

### **11.6. Verify ONNX-PyTorch Support:**

#### **Check ONNX-PyTorch Version:**

``` 
python3 -c "import onnx_torch; print(onnx_torch.__version__)"
```

**Expected Result:**  
- This will print the installed `onnx-torch` version, confirming that ONNX models can be converted to PyTorch format.

---

### **11.7. Additional Verification: TensorFlow-TensorRT Integration:**

#### **Check TensorFlow-TensorRT Installation:**

``` 
python3 -c "import tensorflow as tf; from tensorflow.python.compiler.tensorrt import trt_convert; print(trt_convert)"
```

**Expected Result:**  
- If TensorFlow-TensorRT is installed and properly configured, this will print the `trt_convert` module details, confirming integration.

#### **Test TensorFlow-TensorRT Model Conversion:**

To test TensorFlow-TensorRT integration, you can try converting a simple model:

``` 
python3 -c "import tensorflow as tf; model = tf.keras.applications.MobileNetV2(weights='imagenet'); model.save('mobilenet_v2')"
python3 -c "from tensorflow.python.compiler.tensorrt import trt_convert; converter = trt_convert.TrtGraphConverterV2(input_saved_model_dir='mobilenet_v2'); converter.convert()"
```

**Expected Result:**  
- The model should be successfully converted to TensorRT format, and no errors should occur during the process.

---

### **11.8. Verify Installed Python Packages:**

To ensure all necessary Python packages are installed, use the following command to list them:

``` 
pip3 list
```

**Expected Result:**  
- This will list all installed Python packages. Ensure that `tensorflow`, `torch`, `tensorrt`, `onnx`, `keras`, and other required libraries appear in the list.

---

### Summary of Expected Results:

1. **NVIDIA GPU and CUDA Installation**: Successful detection of GPU and CUDA version.
2. **TensorFlow Installation**: GPU should be listed in `tf.config.list_physical_devices('GPU')`.
3. **PyTorch Installation**: `torch.cuda.is_available()` should return `True`.
4. **TensorRT Installation**: `import tensorrt as trt; print(trt.__version__)` should print the TensorRT version.
5. **ONNX and ONNX-TF/PyTorch**: Version numbers for both `onnx`, `onnx-tf`, and `onnx-torch` should be printed successfully.
