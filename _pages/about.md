---
permalink: /
title: ""
excerpt: ""
author_profile: true
redirect_from: 
  - /about/
  - /about.html
---

{% if site.google_scholar_stats_use_cdn %}
{% assign gsDataBaseUrl = "https://cdn.jsdelivr.net/gh/" | append: site.repository | append: "@" %}
{% else %}
{% assign gsDataBaseUrl = "https://raw.githubusercontent.com/" | append: site.repository | append: "/" %}
{% endif %}
{% assign url = gsDataBaseUrl | append: "google-scholar-stats/gs_data_shieldsio.json" %}

<span class='anchor' id='about-me'></span>

I'm a **Research Fellow at [Microsoft Research](https://www.microsoft.com/en-us/research/lab/microsoft-research-india/) India** (AI4Code). I work on large language models for code — improving how they reason, follow instructions, and edit real-world codebases — across the full stack: training and reinforcement-learning algorithms, multi-GPU distributed training, GPU kernels, and formal-verification tooling for C++.

Alongside research, I build open-source numerical-computing infrastructure. I maintain [`numpy-quaddtype`](https://github.com/SwayamInSync/numpy-quaddtype), a cross-platform 128-bit (quad-precision) floating-point data type for NumPy with 50k+ downloads, as part of [Quansight Labs](https://labs.quansight.org/). Earlier I contributed to [StarCoder](https://arxiv.org/abs/2305.06161) and [OctoPack](https://arxiv.org/abs/2308.07124) with the [BigCode](https://www.bigcode-project.org/) community, and I'm a Kaggle Competition Expert.

My interests sit where machine learning meets systems — making models more capable, and the software they run on faster and more correct. I'm always happy to talk about code LLMs, low-precision numerics, or GPU programming.

You can find my publications on **[Google Scholar](https://scholar.google.com/citations?user=clLJfm8AAAAJ)** <a href='https://scholar.google.com/citations?user=clLJfm8AAAAJ'><img style="vertical-align: middle;" src="https://img.shields.io/endpoint?url={{ url | url_encode }}&logo=Google%20Scholar&labelColor=f6f6f6&color=9cf&style=flat&label=citations"></a>.


# 💼 Experience

- **Research Fellow**, [Microsoft Research](https://www.microsoft.com/en-us/research/lab/microsoft-research-india/) India · *Jul 2024 – Present*<br>Training and adapting LLMs for code generation, editing, and reasoning; multi-GPU distributed training and RL fine-tuning. Lead author of [NextCoder](https://www.microsoft.com/en-us/research/publication/nextcoder-robust-adaptation-of-code-lms-to-diverse-code-edits/) (ICML 2025).
- **Open Source Engineer (NumPy)**, [Quansight Labs](https://labs.quansight.org/) · *Jul 2024 – Present*<br>Author and maintainer of [`numpy-quaddtype`](https://github.com/SwayamInSync/numpy-quaddtype), a cross-platform 128-bit quad-precision dtype (50k+ downloads) built on NumPy's new C DType API.
- **Open Source Research Engineer**, [BigCode](https://www.bigcode-project.org/) · *Feb 2023 – 2024*<br>Contributed to [StarCoder](https://arxiv.org/abs/2305.06161) (15.5B parameters, 1T tokens) and [OctoPack](https://arxiv.org/abs/2308.07124) for instruction tuning of code models.
- **Machine Learning Engineer Intern**, dataX.ai (CrowdANALYTX) · *May 2022 – Nov 2022*<br>Deep-learning models for vision and language; built an ONNX conversion API and custom CUDA kernels for a 2× segmentation speedup.
- **Data Science Intern**, Scaler (InterviewBit) · *2022*<br>Built predictive models and data-preprocessing automation, improving user engagement by ~25%.
- **Applied ML Instructor**, Bili Consultancy · *Jan 2022 – Apr 2022*<br>Mentored undergraduate students in applied machine learning.

# 📝 Publications

- **[NextCoder: Robust Adaptation of Code LMs to Diverse Code Edits](https://www.microsoft.com/en-us/research/publication/nextcoder-robust-adaptation-of-code-lms-to-diverse-code-edits/)** — *ICML 2025*<br><em>Tushar Aggarwal, <strong>Swayam Singh</strong>, Abhijeet Awasthi, Aditya Kanade, Nagarajan Natarajan</em><br>A synthetic-data pipeline and the SeleKT adaptation algorithm that make code LLMs robust to diverse, real-world code edits. &nbsp;[Paper](https://www.microsoft.com/en-us/research/publication/nextcoder-robust-adaptation-of-code-lms-to-diverse-code-edits/) · [Code](https://github.com/microsoft/NextCoder)
- **[Narrow Transformer: StarCoder-Based Java-LM For Desktop](https://arxiv.org/abs/2407.03941)** — *arXiv 2024*<br><em><strong>Swayam Singh</strong>, Karthik Rathinasamy, et al.</em><br>A compact, Java-specialized code LM designed to run on desktop hardware. &nbsp;[arXiv](https://arxiv.org/abs/2407.03941)
- **[OctoPack: Instruction Tuning Code Large Language Models](https://arxiv.org/abs/2308.07124)** — *ICLR 2024 (Spotlight, top 5%)*<br><em>Niklas Muennighoff, … <strong>Swayam Singh</strong>, et al.</em><br>Instruction tuning of code models using natural-language Git commits. &nbsp;[arXiv](https://arxiv.org/abs/2308.07124) · [Code](https://github.com/bigcode-project/octopack)
- **[StarCoder: May the Source Be With You!](https://arxiv.org/abs/2305.06161)** — *TMLR 2023*<br><em>Raymond Li, … <strong>Swayam Singh</strong>, et al. (BigCode)</em><br>A 15.5B-parameter open code LLM trained on 1T tokens of permissively licensed code. &nbsp;[arXiv](https://arxiv.org/abs/2305.06161) · [Code](https://github.com/bigcode-project/starcoder)

# 🚀 Projects

- **[numpy-quaddtype](https://github.com/SwayamInSync/numpy-quaddtype)** — Cross-platform 128-bit quad-precision floating-point dtype for NumPy (50k+ downloads), built on the new NumPy C DType API. *(C · Python)*
- **[QBLAS](https://github.com/SwayamInSync/QBLAS)** — High-performance BLAS for IEEE-754 binary128 (quad) precision. *(C)*
- **[cpp-verify](https://github.com/SwayamInSync/cpp-verify)** — Extending C++ with program-verification constructs backed by SMT solvers, built on LLVM. *(LLVM · C++)*
- **MIRA** — Multimodal Image Reconstruction with Attention: a transformer-based 2D-to-3D reconstruction tool. *(PyTorch)*
- **[Clothes Virtual Try-On](https://github.com/SwayamInSync/clothes-virtual-try-on)** — A ViTON-based virtual clothing try-on assistant (275+ ⭐). *(PyTorch)*
- **MAMBA: Zero to Hero** — Invited talk on State Space Models at [Cohere for AI](https://cohere.com/research) *(Mar 2024)*.

# 📖 Education

- **B.Tech**, University of Allahabad, India · *2020 – 2024*<br>Coursework across data structures, algorithms, operating systems, and big data; focus on machine learning with NLP and computer vision.

# 🏆 Honors & Awards

- **Kaggle Competition Expert** — Bronze medal (top 7%) in the UBC-OCEAN competition; top 3% in the *30 Days of ML* challenge.
- **OctoPack** accepted as a **Spotlight (top 5%)** at ICLR 2024.
- Selected for **Amazon ML Summer School 2023**.
