# Model Zoo
Our plug-and-play Dense Connector module can be easily integrated into existing MLLMs. 
We incorporate the Dense Connector into [Llava-1.5](https://github.com/haotian-liu/LLaVA), [Mini-Gemini](https://github.com/dvlab-research/MGM) (MGM) and [LLaVA-1.6 (LLaVA-NEXT)](https://llava-vl.github.io/blog/2024-01-30-llava-next/), with the models available below.

<!-- In Desnse connector, we evaluate our models across 19 diverse benchmarks, including 11 image benchmarks and 8 video benchmarks. -->

<!--
## 🔥 Dense Connector with LLaVA-1.6
LLaVA-1.6†: 

To further explore the performance of the dense connector across different architectures
Using only data from llava-1.5, the dense connector surpassed llava-next in several benchmarks.

| Model | Data | Vision Encoder | Res. | LLM | Checkpoint | TextVQA | SQA<sup>I</sup>|  GQA  | MMMU | LLaVA-Bench-Wild | MM-Bench | MM-Vet | MathVista 
|----------|---------|---------|---------|----------|:-----------:|---|---|---|---|---|---|---|---|
| Baseline (LLaVA-1.6) | LLaVA-1.6† | CLIP-L/336px | AnyRes | Vicuna-7B | -- | 64.9 | 70.1 | 64.2 | 35.8 | 81.6 | 67.4 | 43.9 | 34.6
| Dense Connector | LLaVA-1.5 | CLIP-L/336px | AnyRes | Vicuna-7B | Coming Soon | 65.6 | 70.5 | 64.6 | 37.6 | 66.9 | 67.4 | 34.8 | 27.0
| Dense Connector | LLaVA-1.5 | SigLIP-so400m | AnyRes | Vicuna-7B | Coming Soon | 66.5 | 69.3 | 64.8 | 36.3 | 70.7 | 67.2 | 34.8 | 27.0
| Dense Connector | MGM | SigLIP-so400m | AnyRes | Vicuna-7B | Coming Soon | 70.0 | 72.0 | 63.9 | 35.8 | 88.8 | 69.2 | 44.4 | 32.7
-->

## Dense Connector with LLaVA-1.5

The overall architecture consists of a visual encoder, Dense Connector, and LLM.

| Model | Data | Vision Encoder | Res. | LLM | Checkpoint | TextVQA | SQA<sup>I</sup>|  GQA  | MMMU | LLaVA-Bench-Wild | MM-Bench | MM-Vet | MathVista 
|----------|----------|----------|----------|-----------|----------|---|---|---|---|---|---|---|---|
| Baseline (LLaVA-1.5) | LLaVA-1.5 | ViT-L/336px | 336 | Vicuna-7B  | -- | 58.2 | 66.8 | 62.0 | 35.3* | 65.4 | 64.3 | 31.1 | 24.9*
| Dense Connector | LLaVA-1.5 | SigLIP-so400m | 384 | Phi2-2.7B | Coming Soon | 55.8 | 70.3 | 61.5 | 36.6 | 65.1 | 70.5 | 33.8 | 28.2
| Dense Connector | LLaVA-1.5 | ViT-L/336px | 336 | Vicuna-7B | Coming Soon | 59.2 | 69.5 | 63.8 | 34.8 | 66.1 | 66.8 | 32.7 | 26.9
| Dense Connector | LLaVA-1.5 | SigLIP-so400m | 384 | Vicuna-7B | [CKPT_HF](https://huggingface.co/HuanjinYao/DenseConnector-v1.5-7B) | 62.6 | 70.5 | 64.4 | 36.7 | 67.4 | 74.4 | 35.4 | 25.5
| Dense Connector | LLaVA-1.5 | SigLIP-so400m | 384 | Vicuna-13B | [CKPT_HF](https://huggingface.co/HuanjinYao/DenseConnector-v1.5-13B) | 64.7 | 73.0 | 65.4 | 34.3 | 73.6 | 71.4 | 41.6 | 29.6
| Dense Connector | LLaVA-1.5 | SigLIP-so400m | 384 | Llama3-8B-Instruct | [CKPT_HF](https://huggingface.co/HuanjinYao/DenseConnector-v1.5-8B) | 62.2 | 75.2 | 65.1 | 40.4 | 68.8 | 74.4 | 34.6 | 28.6
| Dense Connector<sub>LoRa</sub> | LLaVA-1.5 | SigLip-SO | 384 | Hermes-Yi-34B | Coming Soon | 66.7 | 82.4 | 64.0 | 47.1 | 75.1 | 77.7 | 41.0 | 33.5
| Dense Connector<sub>LoRa</sub> | LLaVA-1.5 | SigLip-SO | 384 | Llama3-70B-Instruct | Coming Soon | 66.0 | 82.4 | 64.0 | 47.0 | 74.5 | 79.4 | 46.1 | 32.9

\* denotes results evaluated using official model.


## Applying Dense Connector to Other Models

| Model | Data | Vision Encoder | Resolution | LLM | Checkpoint 
|----------|---------|---------|---------|----------|-----------
| Dense Connector w/ [MGM](https://github.com/dvlab-research/MGM) | MGM | ViT-L/336 + ConvNext-L | 336+768 | Vicuna-7B | [CKPT_HF](https://huggingface.co/HuanjinYao/DenseConnector-with-mgm-7B) | 

Please note that Dense Connector w/ MGM was trained based on the [MGM](https://github.com/dvlab-research/MGM) codebase. Please replace [mgm_arch.py](https://huggingface.co/HuanjinYao/DenseConnector-with-mgm-7B/blob/main/mgm_arch.py) in MGM to test our model.

<!--| SQA<sup>I</sup> | VQAv2 | GQA | VizWiz | MM-Bench | MM-Bench-CN |MM-Vet | MMMU | MathVista |LLaVA-Bench-Wild | MME | TextVQA | POPE
|----------|---------|---------|----------|-----------|---|---|---|---|---|---|---|---|---|---|---|---|---|


## Video BenchMark

| Model | Data | Resolution | LLM | Checkpoint | SQA<sup>I</sup> | VQAv2 | GQA | VizWiz | MM-Bench | MM-Bench-CN |MM-Vet | MMMU | MathVista |LLaVA-Bench-Wild | MME | TextVQA | POPE 
|----------|---------|---------|----------|-----------|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Dense Connector | LLaVA | 336+ | Vicuna-7B | Coming Soon |  -->
