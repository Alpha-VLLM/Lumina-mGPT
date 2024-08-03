
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
