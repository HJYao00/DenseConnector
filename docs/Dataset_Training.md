# Dataset Preparation & Training

## Dataset
We utilize the publicly available datasets [LLaVA-1.5](https://github.com/haotian-liu/LLaVA) and [Mini-Gemini](https://github.com/dvlab-research/MGM). 
Please prepare the datasets in accordance with the official guidelines.

## Training

Our training consists of two stages: the initial pre-training and the subsequent instruction fine-tuning. 
During the pre-training stage, we freeze the ViT and LLM, only updating the Dense Connector with a total batch size of 256. 
In the instruction tuning stage, we update both the Dense Connector and LLM while keeping the ViT frozen, with a global batch size of 128.

You can reduce the `per_device_train_batch_size` and increase the `gradient_accumulation_steps` accordingly to maintain a consistent global batch size. 
Ensure that always keep the global batch size the same: `per_device_train_batch_size` x `gradient_accumulation_steps` x `num_gpus`.

Adjust values `--mm_dense_connector_type` and `--vision_tower` to modify the instantiation of the Dense Connector and visual encoder. For `--mm_dense_connector_type`, we support options 'sti', 'sci' and 'dci'. For `--vision_tower`, the available options are 'openai/clip-vit-large-patch14-336' and 'google/siglip-so400m-patch14-384'.

- `--mm_dense_connector_type dci`: The instantiation of the Dense Connector.
- `--vision_tower openai/clip-vit-large-patch14-336`: Visual Encoder

### Stage 1: Pre-training

We employ DeepSpeed ZERO-2 for pre-training. Please execute the following command to train the Dense Connector on A100 GPUs:

```
sh scripts/v1_5/pretrain.sh
```

In addition, we support training Vicuna-7B & 13B using 32G V100 GPUs, facilitated by xformers</summary>
Given that flash-attention is not supported on V100 GPUs, you will need to install [xformers](https://github.com/facebookresearch/xformers) first. 
After installing xformers, please run the following command to train on V100 GPUs:

```
sh scripts/v1_5/pretrain_v100.sh
```


### Stage2: Instruction Tuning

We use DeepSpeed ZERO-3 for Visual Instruction Tuning. For the Hermes-2-Yi-34B and Llama-3-70B-Instruct, we utilize LoRA for fine-tuning large language models, and we use full fine-tuning for Phi2-2.7B, Vicuna 7&13B and Llama3-8B-Instruct.

To full fine-tuning models for stage 2, you can run the following command:
```
sh scripts/v1_5/finetune.sh
```


To reduce the training memory, you can refer to the following script to train the model using LoRA:
```
sh scripts/v1_5/finetune_lora.sh
```

You can also use V100 GPUs to instruction fine-tuning the Vicuna 7B & 13B models:
```
sh scripts/v1_5/finetune_v100.sh
```

