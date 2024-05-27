# Model Zoo
Our plug-and-play Dense Connector module can be easily integrated into existing MLLMs. 
We incorporate the Dense Connector into [Llava-1.5](https://github.com/haotian-liu/LLaVA) and [Mini-Gemini](https://github.com/dvlab-research/MGM) (MGM), with the models available below.

The models will be available soon.

<!-- In Desnse connector, we evaluate our models across 19 diverse benchmarks, including 11 image benchmarks and 8 video benchmarks. -->

## Base Method

The overall architecture consists of a visual encoder, Dense Connector, and LLM.

| Model | Data | Vision Encoder | LLM | Checkpoint 
|----------|----------|----------|----------|-----------|
| Dense Connector | LLaVA-1.5 | SigLip-SO | Phi2-2.7B | Coming Soon |
| Dense Connector | LLaVA-1.5 | ViT-L/336 | Vicuna-7B | Coming Soon |
| Dense Connector | LLaVA-1.5 | SigLip-SO | Vicuna-7B | [CKPT_HF](https://huggingface.co/HuanjinYao/DenseConnector-v1.5-7B) | 
| Dense Connector | LLaVA-1.5 | SigLip-SO | Vicuna-13B | Coming soon | 
| Dense Connector | LLaVA-1.5 | SigLip-SO | Llama3-8B-Instruct | [CKPT_HF](https://huggingface.co/HuanjinYao/DenseConnector-v1.5-8B) | 
| Dense Connector<sub>LoRa</sub> | LLaVA-1.5 | SigLip-SO | Hermes-Yi-34B | Coming Soon | 
| Dense Connector<sub>LoRa</sub> | LLaVA-1.5 | SigLip-SO | Llama3-70B-Instruct | Coming Soon |


## Applying Dense Connector to Other Models

| Model | Data | Resolution | LLM | Checkpoint 
|----------|---------|---------|----------|-----------
| Dense Connector w/ [MGM](https://github.com/dvlab-research/MGM) | MGM | 336+768 | Vicuna-7B | [CKPT_HF](https://huggingface.co/HuanjinYao/DenseConnector-with-mgm-7B) | 

Please note that Dense Connector w/ MGM was trained based on the [MGM](https://github.com/dvlab-research/MGM) codebase. Please replace [mgm_arch.py](https://huggingface.co/HuanjinYao/DenseConnector-with-mgm-7B/blob/main/mgm_arch.py) in MGM to test our model.

<!--| SQA<sup>I</sup> | VQAv2 | GQA | VizWiz | MM-Bench | MM-Bench-CN |MM-Vet | MMMU | MathVista |LLaVA-Bench-Wild | MME | TextVQA | POPE
|----------|---------|---------|----------|-----------|---|---|---|---|---|---|---|---|---|---|---|---|---|


## Video BenchMark

| Model | Data | Resolution | LLM | Checkpoint | SQA<sup>I</sup> | VQAv2 | GQA | VizWiz | MM-Bench | MM-Bench-CN |MM-Vet | MMMU | MathVista |LLaVA-Bench-Wild | MME | TextVQA | POPE 
|----------|---------|---------|----------|-----------|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Dense Connector | LLaVA | 336+ | Vicuna-7B | Coming Soon |  -->
