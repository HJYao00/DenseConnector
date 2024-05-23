# Model Zoo
Our plug-and-play Dense Connector module can be easily integrated into existing MLLMs. 
We incorporate the Dense Connector into [Llava-1.5](https://github.com/haotian-liu/LLaVA) and [Mini-Gemini](https://github.com/dvlab-research/MGM) (MGM), with the models available below.

In Desnse connector, we evaluate our models across 19 diverse benchmarks, including 11 image benchmarks and 8 video benchmarks.

## Base Method

| Model | Data | Vision Encoder | LLM | Checkpoint | SQA<sup>I</sup> | VQAv2 | GQA | VizWiz | TextVQA | MM-Bench | MM-Bench-CN |MM-Vet | MMMU<sup>v</sup> | MathVista |LLaVA-Bench-Wild | MME<sup>p</sup> | POPE 
|----------|----------|----------|----------|-----------|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Dense Connector | LLaVA-1.5 | SigLip-SO | Phi2-2.7B | Coming Soon |  |  |  |  | 
| Dense Connector | LLaVA-1.5 | ViT-L/336 | Vicuna-7B | Coming Soon | 69.5 | 79.5 | 63.8 | xx | 59.2 | 66.8 | xx | 32.7 | 34.8 | 26.9 | 66.1 |1486 | 86.6
| Dense Connector | LLaVA-1.5 | SigLip-SO | Vicuna-7B | Coming Soon | 70.5 | 81.2 | 64.4 | 53.3 | 62.6 | 68.4 | 62.4 | 35.4 | 36.7 | 25.5 | 67.4 | 1523 | 85.5
| Dense Connector | LLaVA-1.5 | SigLip-SO | Vicuna-13B | Coming Soon | 
| Dense Connector | LLaVA-1.5 | SigLip-SO | Llama3-8B-Instruct | Coming Soon | 
| Dense Connector<sub>LoRa</sub> | LLaVA-1.5 | SigLip-SO | Hermes-Yi-34B | Coming Soon | 
| Dense Connector<sub>LoRa</sub> | LLaVA-1.5 | SigLip-SO | Llama3-70B-Instruct | Coming Soon | 


## Applying Dense Connector to Other Models

| Model | Data | Resolution | LLM | Checkpoint | SQA<sup>I</sup> | VQAv2 | GQA | VizWiz | MM-Bench | MM-Bench-CN |MM-Vet | MMMU | MathVista |LLaVA-Bench-Wild | MME | TextVQA | POPE 
|----------|---------|---------|----------|-----------|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Dense Connector w/ [MGM](https://github.com/dvlab-research/MGM) | MGM | 336+768 | Vicuna-7B | Coming Soon | 


## Video BenchMark

<!-- | Model | Data | Resolution | LLM | Checkpoint | SQA<sup>I</sup> | VQAv2 | GQA | VizWiz | MM-Bench | MM-Bench-CN |MM-Vet | MMMU | MathVista |LLaVA-Bench-Wild | MME | TextVQA | POPE 
|----------|---------|---------|----------|-----------|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Dense Connector | LLaVA | 336+ | Vicuna-7B | Coming Soon |  -->
