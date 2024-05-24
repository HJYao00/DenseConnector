## Evaluation

In Desnse connector, we evaluate our models across 19 diverse benchmarks, including 11 image benchmarks and 8 video benchmarks. You can download our model weights to conduct tests and view our evaluation results from the [Model Zoo]().

Our evaluation is divided into two parts: image and video assessments. Our image evaluation script is based on [LLaVA](https://github.com/haotian-liu/LLaVA/blob/main/docs/Evaluation.md).
For video evaluation, we adopted the [FreeVA](https://github.com/whwu95/FreeVA) method to extend our model's capabilities into video understanding. 
Our model exhibits exceptional temporal understanding abilities without being exposed to any video data during the training phase.


## Evaluate on Image Benchmarks

Our image evaluation follow the LLaVA guidelines. For a more detailed description of the evaluation process, please refer to the link provided [here](https://github.com/haotian-liu/LLaVA/blob/main/docs/Evaluation.md).

We also provide an evaluation example: if you wish to assess the performance of our model on GQA, you can run the following command:

```
sh scripts/v1_5/eval/gqa.sh
```

Please note that if you want to evaluate the model on MMMU benchmark, you should first unzip the `dc/eval/MMMU.zip` file first.

## Evaluate on Video Benchmarks

Our video evaluation process is divided into two steps: the first step involves generating video prediction results using scripts from [here](https://github.com/HJYao00/DenseConnector/tree/main/scripts/v1_5/eval/video), and the second step involves evaluation on GPT-3.5.

For example, if you want to evaluate the model on MSVD-QA benchmark, you should follow these steps:
### First Step: Generate the video predictions

Run the following command to generate video predictions:
```
sh scripts/v1_5/eval/video/run_qa_msvd.sh
```

We use the `--use_pool` option to reduce the number of tokens, allowing the Dense Connector to process more frames.

Moreover, the upper limit of frames T is determined by the `max_position_embeddings` of the large language model. For example, when using vit-L/336px with pooling (where each frame is downsampled by a factor of two), each frame results in 288 tokens. Therefore, the setting of T should satisfy T*288 < max_position_embeddings.


### Second Step: 
After generating video predictions, we evaluate them using gpt-3.5-turbo-0125 for GPT assessment, noting that different GPT versions yield varying results.

```
sh scripts/v1_5/eval/video/gpt_eval/eval_qa_msvd.sh
```
