# Model Zoo
Our plug-and-play Dense Connector module can be easily integrated into existing MLLMs. 
We incorporate the Dense Connector into [Llava-1.5](https://github.com/haotian-liu/LLaVA), [Mini-Gemini](https://github.com/dvlab-research/MGM) (MGM) and [LLaVA-NeXT (LLaVA-1.6)](https://llava-vl.github.io/blog/2024-01-30-llava-next/), with the models available below.

<!-- In Desnse connector, we evaluate our models across 19 diverse benchmarks, including 11 image benchmarks and 8 video benchmarks. -->


## 🔥 Dense Connector with LLaVA-NeXT (LLaVA-1.6)
Recently, we combined the Dense Connector with a dynamic high-resolution approach (i.e., AnyRes in LLaVA-NeXT) to further explore its effectiveness in high-resolution scenarios. Using only the llava-1.5 dataset, the Dense Connector surpassed LLaVA-NeXT (LLaVA-1.6) on several benchmarks. 

> LLaVA-NeXT† dataset includes 558K pre-training data and 790K instruction-tuning data († indicates that the data has not been released yet).
> LLaVA-1.5 dataset consists of 558K pre-training data and 665K instruction-tuning data.
> MGM dataset contains 1.2M pre-training data and 1.5M instruction-tuning data.

| Model | Data | Vision Encoder | Res. | LLM | Checkpoint | TextVQA | SQA<sup>I</sup>|  GQA  | LLaVA-Bench-Wild | MM-Bench | MM-Vet | MMMU | MathVista 
|----------|---------|---------|---------|----------|:-----------:|---|---|---|---|---|---|---|---|
| [LLaVA-NeXT](https://llava-vl.github.io/blog/2024-01-30-llava-next/) | LLaVA-NeXT† | CLIP-L/336px | AnyRes | Vicuna-7B | -- | 64.9 | 70.1 | 64.2 | 81.6 | 67.4 | 43.9 | 35.8 | 34.6
| [LLaVA-NeXT](https://llava-vl.github.io/blog/2024-01-30-llava-next/) (Baseline) | LLaVA-1.5 | CLIP-L/336px | AnyRes | Vicuna-7B | -- | 64.5 | 69.5 | 64.0 | 68.2 | 67.5 | 33.1 | - | 25.7
| Dense Connector | LLaVA-1.5 | CLIP-L/336px | AnyRes | Vicuna-7B | Coming Soon | 65.6 | 70.5 | 64.6 | 66.9 | 67.4 | 33.7 | 37.6 | 26.2
| Dense Connector | LLaVA-1.5 | SigLIP-so400m | AnyRes | Vicuna-7B | Coming Soon | 66.5 | 69.3 | 64.8 | 70.7 | 67.2 | 34.8 | 36.3| 27.0
| Dense Connector | MGM | SigLIP-so400m | AnyRes | Vicuna-7B | Coming Soon | 70.0 | 72.0 | 63.9 | 88.8 | 69.2 | 44.4 | 35.8 | 32.7


## Dense Connector with LLaVA-1.5

Here, we used llava-1.5 as our baseline, training the model with a fixed resolution and llava-1.5 dataset. \* denotes results evaluated using official checkpoints.

| Model | Data | Vision Encoder | Res. | LLM | Checkpoint | TextVQA | SQA<sup>I</sup>|  GQA | LLaVA-Bench-Wild | MM-Bench | MM-Vet | MMMU | MathVista 
|----------|----------|----------|----------|-----------|----------|---|---|---|---|---|---|---|---|
| Baseline ([LLaVA-1.5](https://arxiv.org/abs/2310.03744)) | LLaVA-1.5 | ViT-L/336px | 336 | Vicuna-7B  | -- | 58.2 | 66.8 | 62.0 | 65.4 | 64.3 | 31.1 | 35.3* | 24.9*
| Dense Connector | LLaVA-1.5 | SigLIP-so400m | 384 | Phi2-2.7B | Coming Soon | 55.8 | 70.3 | 61.5 | 65.1 | 70.5 | 33.8 | 36.6 | 28.2
| Dense Connector | LLaVA-1.5 | ViT-L/336px | 336 | Vicuna-7B | Coming Soon | 59.2 | 69.5 | 63.8 | 66.1 | 66.8 | 32.7 | 34.8 | 26.9
| Dense Connector | LLaVA-1.5 | SigLIP-so400m | 384 | Vicuna-7B | [CKPT_HF](https://huggingface.co/HuanjinYao/DenseConnector-v1.5-7B) | 62.6 | 70.5 | 64.4 | 67.4 | 68.4 | 35.4 | 36.7  | 25.5
| Dense Connector | LLaVA-1.5 | SigLIP-so400m | 384 | Vicuna-13B | [CKPT_HF](https://huggingface.co/HuanjinYao/DenseConnector-v1.5-13B) | 64.7 | 73.0 | 65.4 | 73.6 | 71.4 | 41.6 | 34.3| 29.6
| Dense Connector | LLaVA-1.5 | SigLIP-so400m | 384 | Llama3-8B-Instruct | [CKPT_HF](https://huggingface.co/HuanjinYao/DenseConnector-v1.5-8B) | 62.2 | 75.2 | 65.1 | 68.8 | 74.4 | 34.6 | 40.4 | 28.6
| Dense Connector<sub>LoRa</sub> | LLaVA-1.5 | SigLIP-so400m | 384 | Hermes-Yi-34B | Coming Soon | 66.7 | 80.5 | 63.9 | 75.1 | 77.7 | 41.0 | 47.1 | 33.5
| Dense Connector<sub>LoRa</sub> | LLaVA-1.5 | SigLIP-so400m | 384 | Llama3-70B-Instruct | Coming Soon | 66.0 | 82.4 | 64.0 | 74.5 | 79.4 | 46.1 | 47.0 | 32.9



## Applying Dense Connector to Other Models

### Dense Connector with Mini-Gemini
| Model | Data | Vision Encoder | Res. | LLM | Checkpoint | TextVQA | SQA<sup>I</sup>|  GQA  | MMMU | MM-Bench | MM-Vet | MathVista 
|----------|---------|---------|---------|----------|:-----------:|---|---|---|---|---|---|---|
| Baseline (MGM) | MGM | ViT-L/336px + ConvNext-L | 336+768 | Vicuna-7B | -- | 65.2 | 60.4 | 62.6 | 36.1 | 69.3 | 40.8 | 31.4
| Dense Connector w/ [MGM](https://github.com/dvlab-research/MGM) | MGM | ViT-L/336px + ConvNext-L | 336+768 | Vicuna-7B | [CKPT_HF](https://huggingface.co/HuanjinYao/DenseConnector-with-mgm-7B) | 66.0 | 70.7 | 63.3 | 36.8 | 70.7 | 42.2 | 32.5

Please note that Dense Connector w/ MGM was trained based on the [MGM](https://github.com/dvlab-research/MGM) codebase. Please replace [mgm_arch.py](https://huggingface.co/HuanjinYao/DenseConnector-with-mgm-7B/blob/main/mgm_arch.py) in MGM to test our model.

<!--| SQA<sup>I</sup> | VQAv2 | GQA | VizWiz | MM-Bench | MM-Bench-CN |MM-Vet | MMMU | MathVista |LLaVA-Bench-Wild | MME | TextVQA | POPE
|----------|---------|---------|----------|-----------|---|---|---|---|---|---|---|---|---|---|---|---|---|


## Video BenchMark

| Model | Data | Resolution | LLM | Checkpoint | SQA<sup>I</sup> | VQAv2 | GQA | VizWiz | MM-Bench | MM-Bench-CN |MM-Vet | MMMU | MathVista |LLaVA-Bench-Wild | MME | TextVQA | POPE 
|----------|---------|---------|----------|-----------|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Dense Connector | LLaVA | 336+ | Vicuna-7B | Coming Soon |  -->
