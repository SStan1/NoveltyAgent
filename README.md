# NoveltyAgent

NoveltyAgent is the official codebase for our paper:

> **NoveltyAgent: Autonomous Novelty Reporting Agent with Point-wise Novelty Analysis and Self-Validation**

It is designed to help users quickly understand **how novel a research paper is** by retrieving related literature, comparing the paper's main contributions with prior work, and automatically generating a structured novelty analysis report.

⭐ If you find this project useful, please consider giving us a **star** — it helps others discover the project and motivates continued development!

## Overview

![NoveltyAgent Workflow](./Setup/main_figure_new_withexample.png)

Given a research paper, NoveltyAgent will automatically analyze the main contributions of the paper, retrieve relevant related work, compare the paper against prior literature, and generate a structured **novelty report**.

The output helps users quickly understand what the paper mainly proposes, which parts are similar to existing work, which parts are potentially novel, and an overall novelty judgment.

## Core Advantages

Compared with general-purpose review or deep research systems, NoveltyAgent is designed specifically for **paper novelty analysis**. Its main advantages include the following:

- **Point-by-point novelty analysis** — It decomposes a paper into individual novelty points and retrieves evidence for each one separately, enabling a more comprehensive analysis, instead of treating the entire paper as a single query.

- **Full-text literature grounding** — It builds its reference database from the full text of related papers rather than relying only on abstracts, providing richer details for comparison and helping avoid shallow or overly generic analysis.

- **Better faithfulness** — It includes a self-validation step to reduce unsupported claims and improve the reliability of the generated report.

## Demo

We provide a short demo video to illustrate the usage of NoveltyAgent:

https://github.com/user-attachments/assets/89c18aa2-4809-4256-8336-4d0ed74844e1

> **Note:** This demo is for demonstration purposes only. In real-world usage, the analysis process takes significantly longer than shown in the video, as it involves extensive literature retrieval, comparison, and validation steps.

## Installation & Setup

### Prerequisites

- Python 3.9+
- Docker & Docker Compose
- Git
- NVIDIA GPU with CUDA support (for RAGFlow GPU mode and Reranker inference)

### Step 1: Clone the Repository

```bash
git clone https://github.com/SStan1/NoveltyAgent.git
cd NoveltyAgent
```

### Step 2: Install Python Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Download the Reranker Model

Download the **Qwen3-Reranker-4B** model from HuggingFace using `huggingface_hub`:

```bash
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='dengcao/Qwen3-Reranker-4B',
    local_dir='./dengcao/Qwen3-Reranker-4B'
)
"
```

### Step 4: Deploy RAGFlow

#### 4.1 Check and Configure `vm.max_map_count`

Elasticsearch requires `vm.max_map_count` to be at least **262144**. Check the current value:

```bash
sysctl vm.max_map_count
```

If the value is less than 262144, update it:

```bash
sudo sysctl -w vm.max_map_count=262144
```

To make this change permanent, add or update the following line in `/etc/sysctl.conf`:

```
vm.max_map_count=262144
```

#### 4.2 Clone RAGFlow

Clone the RAGFlow repository into your working directory:

```bash
git clone https://github.com/infiniflow/ragflow.git
```

#### 4.3 Replace Configuration Files

Replace RAGFlow's default `.env` and `docker-compose-base.yml` with the customized versions provided in this project's `Setup/` directory:

```bash
cp Setup/.env ragflow/docker/.env
cp Setup/docker-compose-base.yml ragflow/docker/docker-compose-base.yml
```

### Step 5: Start Docker Services

#### 5.1 Start RAGFlow

```bash
cd ragflow/docker
docker compose -f docker-compose-gpu.yml up -d
cd ../..
```

#### 5.2 Start the Reranker Service

Navigate to the Reranker model directory and start the service:

```bash
cd dengcao/Qwen3-Reranker-4B
docker compose up -d
cd ../..
```

### Step 6: Configure the Reranker in RAGFlow

After both Docker services are up and running, you need to **manually** register the Reranker model inside RAGFlow's web UI:

1. Open your browser and navigate to `http://localhost:9380`.
2. Log in to the RAGFlow admin panel.
3. Go to **Model Providers** settings.
4. Add a new **Rerank model** with the following settings:
   - **Model type:** Reranker
   - **Model name:** `Qwen3-Reranker-4B`
   - **Provider / Inference backend:** VLLM
   - Point the model URL to the Reranker service endpoint launched in Step 5.2.
5. Save the configuration.

### Step 7: Configure API Keys

Before running the system, configure the required API keys and related environment variables.

### Step 8: Launch the Application

Start the Streamlit frontend from the project root:

```bash
streamlit run NoveltyAgent/app.py
```

The application will be available by default at:

```
http://localhost:8501
```

## Citation

If you find our work helpful, please consider citing our paper:

```bibtex
@misc{hou2026noveltyagentautonomousnoveltyreporting,
      title={NoveltyAgent: Autonomous Novelty Reporting Agent with Point-wise Novelty Analysis and Self-Validation}, 
      author={Jiajun Hou and Hexuan Deng and Wenxiang Jiao and Xuebo Liu and Xiaopeng Ke and Min Zhang},
      year={2026},
      eprint={2603.20884},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2603.20884}, 
}
```
