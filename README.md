# Ubuntu Setup for Deep Learning and Machine Learning on NVIDIA Systems

This guide provides an **extremely in-depth**, step-by-step walkthrough for configuring an Ubuntu system for deep learning and machine learning tasks using NVIDIA hardware. All installations are performed system-wide (i.e., no virtual environments), so please be aware that these changes affect your entire system. It is recommended to back up your system or create a restore point before proceeding.

> **Disclaimer:** Installing packages system-wide may lead to conflicts or stability issues over time. This guide is intended for users who prefer or require a global setup.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [System Update and Upgrade](#system-update-and-upgrade)
3. [NVIDIA Drivers and CUDA Toolkit](#nvidia-drivers-and-cuda-toolkit)
4. [Global Python Installation and Essential Packages](#global-python-installation-and-essential-packages)
5. [Machine Learning and Data Processing Libraries](#machine-learning-and-data-processing-libraries)
6. [Deep Learning Frameworks](#deep-learning-frameworks)
    - [TensorFlow (GPU-enabled)](#tensorflow-gpu)
    - [PyTorch (with CUDA support)](#pytorch-cuda)
    - [TensorFlow-TensorRT Integration](#tensorflow-tensorrt-integration)
    - [PyTorch-TensorRT Integration](#pytorch-tensorrt-integration)
7. [cuDNN Installation](#cudnn-installation)
8. [TensorRT Installation](#tensorrt-installation)
9. [Profiling and Debugging Tools](#profiling-and-debugging-tools)
10. [Verification and Testing](#verification-and-testing)
11. [Troubleshooting and Further Resources](#troubleshooting-and-further-resources)

---

## 1. Prerequisites

- **Ubuntu Versions Tested:** Ubuntu 20.04 and Ubuntu 22.04.
- **Hardware:** An NVIDIA GPU with supported drivers.
- **Internet Connection:** Required for downloading packages.
- **Administrator Access:** You will need `sudo` privileges to install system-wide packages.

**Official Resources:**
- [NVIDIA Driver Downloads](https://www.nvidia.com/Download/index.aspx)
- [Ubuntu Documentation](https://help.ubuntu.com/)

---

## 2. System Update and Upgrade

Ensure your system is up-to-date. This minimizes conflicts and ensures all dependencies are current.

```bash
sudo apt update && sudo apt upgrade -y
```

---

## 3. NVIDIA Drivers and CUDA Toolkit

### 3.1. Install NVIDIA Drivers

Install the latest NVIDIA drivers using Ubuntu’s driver management tool. This ensures compatibility with CUDA and your GPU hardware.

```bash
sudo ubuntu-drivers autoinstall
```

After installation, reboot your system:

```bash
sudo reboot
```

### 3.2. Install CUDA Toolkit

The CUDA toolkit provides the necessary libraries and tools to run GPU-accelerated applications.

**Option A: Install via Ubuntu Repositories**

```bash
sudo apt install nvidia-cuda-toolkit
```

**Option B: Download from NVIDIA**

For the latest version and more control over the installation, visit the [NVIDIA CUDA Toolkit Downloads](https://developer.nvidia.com/cuda-downloads) page and follow the instructions for your Ubuntu version.

**Verify Installation:**

```bash
nvcc --version
nvidia-smi
```

Expected output:
- `nvcc --version` should display the CUDA compiler version.
- `nvidia-smi` should list your GPU details along with driver and CUDA versions.

---

## 4. Global Python Installation and Essential Packages

### 4.1. Install Python 3 and pip

Install Python 3 and its package manager globally:

```bash
sudo apt install python3 python3-pip
```

### 4.2. Upgrade pip Globally

Keep pip updated to avoid compatibility issues:

```bash
sudo -H pip3 install --upgrade pip
```

### 4.3. Install Utility Packages

Install some utilities to improve your Python experience (e.g., enhanced error messages):

```bash
sudo -H pip3 install pretty_errors
```

Test the installation:

```bash
python3 -m pretty_errors
```

---

## 5. Machine Learning and Data Processing Libraries

Install common libraries used in data processing and machine learning:

```bash
sudo -H pip3 install numpy scipy pandas matplotlib seaborn scikit-image pillow opencv-python-headless
```

*Note:* We install OpenCV via pip in its headless version to avoid GUI dependencies on servers.

---

## 6. Deep Learning Frameworks

### TensorFlow (GPU-enabled)
Install TensorFlow with GPU support:

```bash
sudo -H pip3 install tensorflow
```

For the most recent instructions and compatibility details, refer to the [TensorFlow Installation Guide](https://www.tensorflow.org/install).

### PyTorch (with CUDA support)
Install PyTorch using the recommended installation command for your CUDA version. For example, for CUDA 11.7, visit the [PyTorch Get Started](https://pytorch.org/get-started/locally/) page. A typical command might be:

```bash
sudo -H pip3 install torch torchvision
```

### TensorFlow-TensorRT Integration

Enable integration between TensorFlow and TensorRT for optimized inference:

```bash
sudo -H pip3 install tensorflow-tensorrt
```

### PyTorch-TensorRT Integration

Install PyTorch-TensorRT for accelerated PyTorch inference:

```bash
sudo -H pip3 install torch-tensorrt
```

---

## 7. cuDNN Installation

cuDNN is essential for deep learning performance and must match your CUDA version.

1. **Download cuDNN:**
   - Visit the [NVIDIA cuDNN Download page](https://developer.nvidia.com/cudnn) (login required).
   - Select the version that matches your installed CUDA toolkit.
2. **Installation Instructions:**
   - Follow the official guide provided with the download package. Typically, you will extract the files and copy them to the appropriate CUDA directories (e.g., `/usr/local/cuda/include` and `/usr/local/cuda/lib64`).

*Refer to the [cuDNN Installation Guide](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html) for detailed steps.*

---

## 8. TensorRT Installation

TensorRT accelerates inference on NVIDIA GPUs.

### 8.1. Download TensorRT

Download the TensorRT package for your Ubuntu version from the [NVIDIA TensorRT Download page](https://developer.nvidia.com/tensorrt). For Ubuntu 20.04, an example download command is:

```bash
sudo wget https://developer.download.nvidia.com/compute/machine-learning/tensorrt/8.6.1.6/ubuntu2004/x86_64/tensorrt-repo-ubuntu2004-8.6.1.6-ga-cuda11.7_1-1_amd64.deb
```

### 8.2. Install TensorRT Repository and Keys

```bash
sudo dpkg -i tensorrt-repo-ubuntu2004-8.6.1.6-ga-cuda11.7_1-1_amd64.deb
sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/machine-learning/tensorrt/8.6.1.6/ubuntu2004/x86_64/7fa2af80.pub
sudo apt update
```

### 8.3. Install TensorRT Libraries

```bash
sudo apt install libnvinfer8 libnvinfer-dev libnvinfer-plugin8
```

### 8.4. Install TensorRT Python Bindings

```bash
sudo -H pip3 install nvidia-pyindex tensorrt
```

**Verify TensorRT Installation:**

```bash
python3 -c "import tensorrt as trt; print(trt.__version__)"
```

---

## 9. Profiling and Debugging Tools

Install NVIDIA’s profiling and debugging tools to optimize and troubleshoot GPU applications:

### 9.1. Nsight Systems and Nsight Compute

```bash
sudo apt install nsight-systems nsight-compute
```

These tools provide detailed insights into GPU performance and can help diagnose issues in deep learning workloads.

---

## 10. Verification and Testing

After installing all components, verify that your system is correctly set up.

### 10.1. Verify NVIDIA and CUDA Setup

```bash
nvidia-smi
nvcc --version
```

### 10.2. Verify TensorFlow GPU Support

```bash
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

Expected output: A list showing your GPU(s).

### 10.3. Verify PyTorch GPU Support

```bash
python3 -c "import torch; print(torch.cuda.is_available())"
```

Expected output: `True`

### 10.4. Verify TensorRT Integration

```bash
python3 -c "import tensorrt as trt; print(trt.__version__)"
python3 -c "import tensorflow as tf; from tensorflow.python.compiler.tensorrt import trt_convert; print(trt_convert)"
```

### 10.5. Verify ONNX (Optional)

If you require ONNX for model interoperability, install and verify as follows:

```bash
sudo -H pip3 install onnx onnx-tf onnx-torch
python3 -c "import onnx; print(onnx.__version__)"
python3 -c "import onnx_tf; print(onnx_tf.__version__)"
python3 -c "import onnx_torch; print(onnx_torch.__version__)"
```

---

## 11. Troubleshooting and Further Resources

### Common Issues and Tips

- **Driver or CUDA Mismatch:**  
  Ensure that the installed NVIDIA driver is compatible with your CUDA toolkit. Consult the [CUDA Compatibility Guide](https://docs.nvidia.com/deploy/cuda-compatibility/index.html) for details.

- **cuDNN Version Mismatch:**  
  The cuDNN version must align with your CUDA version. Double-check the compatibility matrix on the [cuDNN Release Notes](https://docs.nvidia.com/deeplearning/cudnn/release-notes/index.html).

- **System-wide Python Conflicts:**  
  Since this guide avoids virtual environments, conflicts might arise from global package installations. Regularly update and manage packages carefully.

- **TensorRT Issues:**  
  Refer to the [TensorRT Documentation](https://docs.nvidia.com/deeplearning/tensorrt/) for troubleshooting integration issues and performance tuning.

### Official Documentation and Community Resources

- **CUDA Toolkit:** [NVIDIA CUDA Documentation](https://docs.nvidia.com/cuda/)
- **cuDNN:** [cuDNN Installation Guide](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html)
- **TensorFlow:** [TensorFlow Installation Guide](https://www.tensorflow.org/install)
- **PyTorch:** [PyTorch Get Started](https://pytorch.org/get-started/locally/)
- **TensorRT:** [TensorRT Documentation](https://docs.nvidia.com/deeplearning/tensorrt/)
- **Nsight Tools:** [Nsight Systems Documentation](https://developer.nvidia.com/nsight-systems) and [Nsight Compute Documentation](https://developer.nvidia.com/nsight-compute)

---

This guide is designed to provide a robust, system-wide setup for deep learning and machine learning on Ubuntu with NVIDIA hardware. For additional support, refer to the official documentation linked above or open an issue in this repository.
