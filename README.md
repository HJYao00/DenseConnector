<div align="center">

<h1> Dense Connector for MLLMs </h1>

<h5 align="center"> 

<a href=''><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>
<a href=''><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-blue'></a>

[Huanjin Yao<sup>*</sup>](https://scholar.google.com/citations?user=pDtsCBQAAAAJ&hl=zh-CN),
[Wenhao Wu<sup>*</sup>](https://whwu95.github.io/),
[Taojiannan Yang](),
[Yuxin Song](),
[Mengxi Zhang](),
[Haocheng Feng](),
[Yifan Sun](),

[Zhiheng Li](https://www.sigs.tsinghua.edu.cn/lzh/main.htm),
[Wanli Ouyang](https://wlouyang.github.io/),
[Jingdong Wang](https://jingdongwang2017.github.io/),

</h5>
</div>

## News
- [x] **[5/24]** We relase **Dense Connector** in [arxiv]()! The code and model will be available soon.


## Contents
- [Overview](#overview)
- [Installation](#installation)
- [Model Zoo](#model-zoo)
- [Dataset Preparation & Training](#dataset-preparation-and-training)
- [Evaluation](#evaluation)
- [Citation](#citation)
- [Acknowledgement](#acknowledgment)


## Overview

<div align=center>
<img width="795" alt="image" src="images/teaser.jpg">
</div>

The Dense Connector utilizes multi-layer visual features to enhance visual representation and augment the visual perception capabilities of the Multimodal Large Language Models (MLLMs) which can be easily integrated into the current MLLMs. We provide three instantiation methods of Dense Connector: Sparse Token Integration (STI), Sparse Channel Integration (SCI), and Dense Channel Integration (DCI). The Dense Channel Integration achieves the best results.

<div align=center>
<img width="795" alt="image" src="images/main.jpg">
</div>

## Installation
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

## Dataset Preparation and Training
Please refer to the [document](https://github.com/HJYao00/DenseConnector/blob/main/docs/Dataset_Training.md) for dataset preparation and training.

## Evaluation
We evaluate the Dense Connector across 19 diverse benchmarks, including 11 image benchmarks and 8 video benchmarks. The testing procedures for both images and videos can be found [here](https://github.com/HJYao00/DenseConnector/blob/main/docs/Evaluation.md).

## Model Zoo
Please visit our [Model Zoo](https://github.com/HJYao00/DenseConnector/blob/main/docs/ModelZoo.md) to access all publicly available Dense Connector checkpoints. 
We scale the LLM from 2.7B to 70B, incorporating the latest open-source large language model, Llama3-8B-Instruct & Llama3-70B-Instruct

## Dialogue Example

We provide several dialogue examples, with additional results available in the [paper]().

<div align=center>
<img width="530" alt="image" src="images/qualitative_results.jpg">
</div>

## Citation
If you find this repository is useful, please consider star🌟 this repo and cite🖇️ our paper.
```bibtex

```

## Acknowledgment

We extend our gratitude to the open-source efforts of [LLaVA](https://github.com/haotian-liu/LLaVA), [Mini-Gemini](https://github.com/dvlab-research/MGM) and [FreeVA](https://github.com/whwu95/FreeVA).
