# Experiments

Here is a personal record of experiments I've run to improve the performance of k2quant. Each of these measures PPL on the WikiText2 dataset and quantization time. Qwen1.5-MoE-A2.7B was used for all experiments. An NVIDIA A100 80GB SXM4 was used for all experiments, although since I rent GPUs from [Vast.ai](https://vast.ai), performance may vary as a function of the individual instance characteristics (CPU, RAM, etc.).

| Implementation                                 | Ablation                | PPL       | Time per layer |
| ---------------------------------------------- | ----------------------- | --------- | -------------- |
| Python VPTQ + FAISS                            | No column ordering      | 8.13      | ~60s           |
| Python VPTQ + FAISS                            | -                       | 7.56      | ~60s           |
| Python VPTQ + FAISS                            | -                       | 7.56      | ~60s           |
| C++ VPTQ                                       | -                       | 7.50      | ~60s           |
| C++ VPTQ + FAISS                               | -                       | 7.88 (?)  | 25s            |
| C++ VPTQ + FAISS (KMeans++)                    | -                       | 7.86 (?)  | 35s            |
| C++ VPTQ + FAISS (KMeans++) + Full sampling\*  | -                       | -         | -              |
| C++ VPTQ (transposed K-means)                  | -                       | 7.48 \*\* | 50s            |
| C++ VPTQ (transposed K-means) + k-factor 1/128 | -                       | 7.53      | 50s            |
| C++ VPTQ (transposed K-means) + k-factor 1/128 | No column ordering      | 7.81???   | 50s            |
| C++ VPTQ (transposed K-means) + k-factor 1/128 | Reverse column ordering | -         | 50s            |

\* Usage of `KMeans.cp.max_points_per_centroid = 0` to force full sampling of centroids. Did not complete in a reasonable amount of time.

\*\* Reason for 0.02 PPL improvement is unknown.
