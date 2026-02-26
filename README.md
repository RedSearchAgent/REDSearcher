<p align="center">
    <img src="figs/red-R.png" width="125" style="margin-bottom: 0.2;"/>
<p>
<h1 align="center"> REDSearcher: A Scalable and Cost-Efficient
Framework for Long-Horizon Search Agents </a></h1>

The official repo for "REDSearcher: A Scalable and Cost-Efficient
Framework for Long-Horizon Search Agents".
<!-- [![arXiv](https://img.shields.io/badge/arXiv-2602.xxxxx-b31b1b.svg)](http://arxiv.org/abs/2602.xxxxx) -->

<p align="center">
<!-- <img src="figs/red-R.png" alt="logo" height="40" align="center" /> -->
📃 <a href="https://redsearchagent.github.io/">Project Page</a>
</p> 


<p align="center">
       🤗 <a href="https://huggingface.co/collections/Zchu/redsearcher">RedSearcher Collections</a>&nbsp&nbsp | &nbsp&nbsp🤗 <a href="https://huggingface.co/datasets/Zchu/REDSearcher_SFT_10K">SFT Dataset (Text)</a>&nbsp&nbsp | &nbsp&nbsp🤗 <a href="https://huggingface.co/datasets/Zchu/REDSearcher_RL_1K">RL Dataset Demo (Text) </a>&nbsp&nbsp | &nbsp&nbsp🤗 <a href="https://huggingface.co/datasets/honglyhly/REDSearcher_MM_SFT_5K">SFT Dataset (MM)</a>
</p>

<p align="center">
       🤗 <a href="https://huggingface.co/collections/Zchu/redsearcher">REDSearcher-30B-A3B (SFT+RL, coming soon)</a>&nbsp&nbsp | &nbsp&nbsp🤗 <a href="https://huggingface.co/collections/Zchu/redsearcher">REDSearcher-MM-30B-A3B (SFT+RL, coming soon)</a>&nbsp&nbsp 
</p>

<p align="center">
  📑<a href="https://arxiv.org/abs/2602.14234v1"> REDSearcher Paper</a>
</p>

We will release the SFT data, RL data, SFT code, RL code, evaluation code, and model weights. Stay tuned for update!

## Timeline
- [2026/02/14] **We released [SFT Dataset (text)](https://huggingface.co/datasets/Zchu/REDSearcher_SFT_10K), [RL Dataset Demo (text)](), [SFT Dataset (MM)](https://huggingface.co/datasets/honglyhly/REDSearcher_MM_SFT_5K)**!


## Overview

**REDSearcher** integrates expert-level query synthesis, capability mid-training, and SFT/RL post-training for scalable search-agent development:

- **Complex Task Synthesis**: Dual-constrained optimization based on graph topology and evidence dispersion
- **Tool-Augmented Queries**: Encouraging proactive tool use over passive recall
- **Mid-Training**: Strengthening core atomic capabilities and agentic capabilities (knowledge, planning, function calling, long-horizon interaction)
- **RL-based Post-Training**: Leveraging a local simulated environment for rapid algorithmic iteration and large-scale reinforcement learning to enhance search performance

## Performance

![](figs/performance.png)

### Quickstart

> **Coming soon** – Detailed quickstart guide will be added in a future update. Stay tuned!

### Environment Setup

> **Coming soon** – Installation instructions and environment setup will be added in a future update.

### Data Processing

#### SFT Data 

Our text-only SFT dataset (10K trajectories) and multi-modal SFT dataset (5K) is available on [Hugging Face](https://huggingface.co/datasets/Zchu/REDSearcher_SFT_10K). The data is formatted in **ShareGPT** format and can be directly used with [ms-swift](https://github.com/modelscope/ms-swift):

```python
from datasets import load_dataset

dataset_text = load_dataset("Zchu/REDSearcher_SFT_10K")
dataset_mm = load_dataset("honglyhly/REDSearcher_MM_SFT_5K")
```

#### RL Data

Our RL dataset is available on [Hugging Face](https://huggingface.co/datasets/Zchu/REDSearcher_RL_1K). To use with [Slime](https://github.com/THUDM/slime), convert the data to the required format where each sample contains `"prompt"` (messages including system prompt) and `"label"` (answer):

```python
from datasets import load_dataset

dataset = load_dataset("Zchu/REDSearcher_RL")
```

### Train

#### SFT 

Our SFT implementation is based on [ms-swift](https://github.com/modelscope/ms-swift) (version 3.12.3) with Megatron-LM for both mid-training and SFT. Please refer to the [ms-swift repository](https://github.com/modelscope/ms-swift) for environment setup and installation instructions.

Below is an example training script for distributed SFT:

```bash
# Example SFT script with ms-swift and Megatron-LM
export MODELSCOPE_CACHE="/path/to/ms_cache"
export MEGATRON_LM_PATH=/path/to/Megatron-LM

PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' NNODES=$WORLD_SIZE NODE_RANK=$RANK megatron sft \
    --model /path/to/pretrained_model \
    --load_safetensors true \
    --save_safetensors true \
    --dataset "/path/to/your_dataset.jsonl" \
    --dataset_shuffle true \
    --tensor_model_parallel_size 2 \
    --expert_model_parallel_size 8 \
    --pipeline_model_parallel_size 2 \
    --context_parallel_size 4 \
    --micro_batch_size 1 \
    --global_batch_size 128 \
    --packing true \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    --max_epochs 2 \
    --save_strategy epoch \
    --finetune true \
    --no_gradient_accumulation_fusion true \
    --cross_entropy_loss_fusion true \
    --lr 3.08e-5 \
    --lr_warmup_fraction 0.05 \
    --min_lr 1.0e-6 \
    --save /path/to/checkpoints \
    --save_retain_interval 3 \
    --max_length 131072 \
    --num_workers 128 \
    --dataset_num_proc 16 \
    --sequence_parallel true \
    --use_flash_attn true \
    --model_type qwen3_moe \
    --loss_scale default
```

#### RL

> **Coming soon** – RL training scripts and instructions will be added in a future update. Our RL implementation is based on [Slime](https://github.com/THUDM/slime).

### Trajectory Synthesis & Evaluation

We use [DeepTraceHub](https://github.com/RedSearchAgent/DeepTraceHub) for trajectory synthesis and model evaluation. 

#### Trajectory Synthesis & Evaluation Overview

DeepTraceHub provides a modular framework for both trajectory synthesis and model evaluation:

1.  **Deployment**: Deploy the Agent Model (e.g., REDSearcher), a Summarizer Model (for web content), and an LLM-as-Judge Model (for automated scoring).
2.  **Configuration**: Set up API keys (Google Search/Serper, Jina, etc.) in a `.env` file and configure agent behavior in a YAML file.
3.  **Trajectory Synthesis**: Generate high-quality training data by running the agent on complex tasks. The framework supports multi-sample generation and various reasoning loops (e.g., ReACT, DeepSeek Thinking-with-Tools).
4.  **Evaluation**: Run automated performance measurement using integrated judges and benchmarks. It supports multi-process execution for high-throughput inference.
5.  **Output**: Results include detailed trajectory JSONs (formatted for SFT/RL training), aggregated statistics, and accuracy summaries.

#### Model Deployment

Before running evaluation or synthesis, you need to deploy the necessary LLM services using [vLLM](https://github.com/vllm-project/vllm) or [SGLang](https://github.com/sgl-project/sglang):

1.  **Agent Model**: 
  - For **Evaluation**: Deploy the target Agent model (e.g., REDSearcher).
  - For **Trajectory Synthesis**: Deploy a strong model with advanced agentic capabilities (e.g., GPT-OSS, DeepSeek-V3.2).
2.  **Summarizer Model**: Deploy a lightweight model (e.g., Qwen3-30B-A3B-Instruct) to summarize retrieved web page content.
3.  **LLM-as-Judge Model**: Deploy a powerful model (e.g., GPT-OSS) to evaluate the agent's final answers against ground truth.


  ```
  vllm serve ${MODEL_PATH} \
    --port ${run_port} \
    --gpu-memory-utilization 0.8 \
    --tensor-parallel-size ${TP_SIZE} \
    --pipeline-parallel-size ${PP_SIZE} \
    --max-model-len ${MAX_MODEL_LEN} \
    --served-model-name ${MODEL_NAME} \
    --trust-remote-code \
    --disable-log-request \
    --enable-prefix-caching \
    --async-scheduling &
  ```

  ```
  python3 -m sglang.launch_server \
      --model-path ${MODEL_PATH} \
      --served-model-name ${MODEL_NAME} \
      --trust-remote-code \
      --host 0.0.0.0 \
      --tool-call-parser ${TOOL_CALL_PARSER} \
      --enable-metrics \
      --max-running-requests ${MAX_RUNNING_REQUESTS} \
      --cuda-graph-max-bs 64 \
      --reasoning-parser ${REASONING_PARSER} \
      --tensor-parallel-size ${TP_SIZE} \
      --pipeline-parallel-size ${PP_SIZE} \
      --context-length ${MAX_MODEL_LEN} \
      --port ${run_port} &
  ```

#### Running Agent Loops via DeepTraceHub

Once the models are deployed, you can use DeepTraceHub to run the agent loops for evaluation or trajectory synthesis:

```bash
cd DeepTraceHub
export LOG_FILE=/path/to/output/logs.log
python src/agent.py \
    --config_path /path/to/config/your_config.yaml \
    --multiprocess \
    --resume
```

For more details, please refer to the [DeepTraceHub README](https://github.com/RedSearchAgent/DeepTraceHub).


## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=RedSearchAgent/REDSearcher&type=date&legend=top-left)](https://www.star-history.com/#RedSearchAgent/REDSearcher&type=date&legend=top-left)


## Citation

```
@article{redsearcher2026,
  title={REDSearcher: A Scalable and Cost-Efficient Framework for Long-Horizon Search Agents},
  author={Zheng Chu and Xiao Wang and Jack Hong and Huiming Fan and Yuqi Huang and Yue Yang and Guohai Xu and Shengchao Hu and Dongdong Kuang and Chenxiao Zhao and Cheng Xiang and Ming Liu and Bing Qin and Xing Yu},
  journal={arXiv preprint arXiv:2602.14234},
  url={https://arxiv.org/pdf/2602.14234},
  year={2026}
}
```
