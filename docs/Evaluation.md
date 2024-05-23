## Evaluation

In Desnse connector, we evaluate our models across 20 diverse benchmarks, including 12 image benchmarks and 8 video benchmarks. You can download our model weights to conduct tests and view our evaluation results from the [Model Zoo]().

Our evaluation is divided into two parts: image and video assessments. Our image evaluation script is based on [LLaVA](https://github.com/haotian-liu/LLaVA/blob/main/docs/Evaluation.md).
For video evaluation, we adopted the [FreeVA](https://github.com/whwu95/FreeVA) method to extend our model's capabilities into video understanding. 
Our model exhibits exceptional temporal understanding abilities without being exposed to any video data during the training phase.


## Evaluate on Image Benchmarks

Our image evaluation follow the LLaVA guidelines. For a more detailed description of the evaluation process, please refer to the link provided [here](https://github.com/haotian-liu/LLaVA/blob/main/docs/Evaluation.md).

We also provide an evaluation example: if you wish to assess the performance of our model on GQA, you can run the following command:

```
sh scripts/v1_5/eval/gqa.sh
```


## Evaluate on Video Benchmarks
Coming Soon!
