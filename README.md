<div align="center">

<h1> Dense Connector for MLLMs </h1>

<h5 align="center"> 
  
</h5>
</div>

<div align=center>
<img width="795" alt="image" src="images/teaser.jpg">
</div>

## 🗺️ Overview

The Dense Connector utilizes multi-layer visual features to enhance visual representation and augment the visual perception capabilities of the Multimodal Large Language Models (MLLMs) which can be easily integrated into the current MLLMs. We provide three instantiation methods of Dense Connector: Sparse Token Integration (STI), Sparse Channel Integration (SCI), and Dense Channel Integration (DCI). The Dense Channel Integration achieves the best results.

<div align=center>
<img width="795" alt="image" src="images/main.jpg">
</div>

## Install
Please follow the instructions below to install the required packages.

1. Clone this repository

2. Install Package
```bash
conda create -n dc python=3.10 -y
conda activate dc
cd DenseConnector
pip install --upgrade pip 
pip install -e .
```

3. Install additional packages for training Dense Connector
```bash
pip install ninja
pip install flash-attn --no-build-isolation
```

## Data
We utilize the publicly available datasets LLaVA-1.5 and Mini-Gemini. Please prepare the datasets in accordance with the official guidelines.
