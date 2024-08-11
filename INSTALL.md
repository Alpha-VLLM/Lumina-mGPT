
### 1. Basic Setup

```
# Create a new conda environment named 'lumina_mgpt' with Python 3.10
conda create -n lumina_mgpt python=3.10 -y
# Activate the 'lumina_mgpt' environment
conda activate lumina_mgpt
# Install required packages from 'requirements.txt'
pip install -r requirements.txt
```

### 2. Install Flash-Attention
```
pip install flash-attn --no-build-isolation
```

### 3. Install xllmx as Python Package
The [xllmx](./xllmx) module is a lightweight engine designed to support the training and inference of
LLM-centered Any2Any models. It is evolved from [LLaMA2-Accessory](https://github.com/Alpha-VLLM/LLaMA2-Accessory), undergoing comprehensive improvements to achieve higher efficiency and
wider functionality, including the support for flexible arrangement and processing of interleaved media and text.

The Lumina-mGPT implementation heavily relies on xllmx and requires xllmx to be installed as a python package (**so that `import xllmx` can be used anywhere in your machine, without the restriction of working directory**).
The installation process is as follows:
```bash
# bash
# go to the root path of the project
cd Lumina_mGPT
# install as package
pip install -e .
```

### 4. Optional: Install Apex
> [!Caution]
>
> If you merely run inference, there is no need to install Apex.
>
> For training, Apex can bring some training efficiency improvement, but it is still not a must.
>
> Note that training works smoothly with either:
> 1. Apex not installed at all; OR
> 2. Apex successfully installed with CUDA and C++ extensions.
>
> However, it will fail when:
> 1. A Python-only build of Apex is installed.
>
> If errors like `No module named 'fused_layer_norm_cuda'` are reported, it generally means that you are
using a Python-only Apex build. Please run `pip uninstall apex` to remove the build and try again.

Lumina-mGPT utilizes [apex](https://github.com/NVIDIA/apex) to accelerate training, which needs to be compiled from source. Please follow the [official instructions](https://github.com/NVIDIA/apex#from-source) for installation.
Here are some tips based on our experiences:

**Step1**: Check the version of CUDA with which your torch is built:
 ```python
# python
import torch
print(torch.version.cuda)
```

**Step2**: Check the CUDA toolkit version on your system:
```bash
# bash
nvcc -V
```
**Step3**: If the two aforementioned versions mismatch, or if you do not have CUDA toolkit installed on your system,
please download and install CUDA toolkit from [here](https://developer.nvidia.com/cuda-toolkit-archive) with version matching the torch CUDA version.

> [!Note]
>
> Note that multiple versions of CUDA toolkit can co-exist on the same machine, and the version can be easily switched by changing the `$PATH` and `$LD_LIBRARY_PATH` environment variables.
There is thus no need to worry about your machine's environment getting messed up.

**Step4**: You can now start installing apex:
```bash
git clone https://github.com/NVIDIA/apex
cd apex
# if pip >= 23.1 (ref: https://pip.pypa.io/en/stable/news/#v23-1) which supports multiple `--config-settings` with the same key...
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
# otherwise
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```
