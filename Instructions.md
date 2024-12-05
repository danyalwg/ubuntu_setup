# **Step-by-Step Ubuntu Setup for Machine Learning**

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
After completing the setup, verify that all components are correctly installed and working.

### **11.1. Verify NVIDIA GPU and CUDA:**

``` 
nvidia-smi
nvcc --version
```

**Expected Result:**  
- `nvidia-smi` should show your GPU details.
- `nvcc --version` should return the CUDA version installed.

### **11.2. Verify TensorFlow GPU Access:**

``` 
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

**Expected Result:**  
The list of available GPUs should be printed, confirming that TensorFlow can access your GPU.

### **11.3. Verify TensorRT Installation:**

``` 
python3 -c "import tensorrt as trt; print(trt.__version__)"
```

**Expected Result:**  
The TensorRT version installed should be displayed.

### **11.4. Verify ONNX Installation:**

``` 
python3 -c "import onnx; print(onnx.__version__)"
```

**Expected Result:**  
The ONNX version installed should be printed.
