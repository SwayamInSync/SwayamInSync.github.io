---
permalink: /
title: ""
excerpt: ""
author_profile: true
redirect_from: 
  - /about/
  - /about.html
---

<span class='anchor' id='about-me'></span>

I'm a **Research Fellow at [Microsoft Research](https://www.microsoft.com/en-us/research/lab/microsoft-research-india/) India** (AI4Code). I work on large language models for code — improving how they reason, follow instructions, and edit real-world codebases — across the full stack: training and reinforcement-learning algorithms, multi-GPU distributed training, GPU kernels, and formal-verification tooling for C++.

Alongside research, I build open-source numerical-computing infrastructure. I author and maintain [`numpy-quaddtype`](https://github.com/SwayamInSync/numpy-quaddtype), a cross-platform 128-bit (quad-precision) floating-point data type for NumPy with 100k+ downloads, as part of [Quansight Labs](https://labs.quansight.org/). Earlier I contributed to [StarCoder](https://arxiv.org/abs/2305.06161) and [OctoPack](https://arxiv.org/abs/2308.07124) with the [BigCode](https://www.bigcode-project.org/) community, and I'm a Kaggle Competition Expert.

My interests sit where machine learning meets systems — making models more capable, and the software they run on faster and more correct. I have <strong><span id='total_cit'>—</span></strong> citations on [Google Scholar](https://scholar.google.com/citations?user=clLJfm8AAAAJ). <!-- the count fills in automatically once the Scholar crawler runs on the deployed repo -->


# 🔥 News
- *2025.05*: &nbsp;🎉 **NextCoder** was accepted at **ICML 2025**. <!-- 🚧 MARKER: verify exact month -->
- *2024.07*: &nbsp;🎉 Joined **Microsoft Research** India as a Research Fellow on the AI4Code team.
- *2024.07*: &nbsp;📦 Released **`numpy-quaddtype`** (quad-precision for NumPy), now with 100k+ downloads.
- *2024.06*: &nbsp;🏅 Reached **Kaggle Competition Expert**.
- *2023.10*: &nbsp;🎉 **OctoPack** accepted as a **Spotlight (top 5%)** at ICLR 2024. <!-- 🚧 MARKER: verify month -->
<!-- 🚧 MARKER: add your latest news items here -->

# 💼 Experience

- **Research Fellow**, [Microsoft Research](https://www.microsoft.com/en-us/research/lab/microsoft-research-india/) India · *Jul 2024 – Present*<br>Training and adapting LLMs for code generation, editing, and reasoning; multi-GPU distributed training and RL fine-tuning. Lead author of [NextCoder](https://www.microsoft.com/en-us/research/publication/nextcoder-robust-adaptation-of-code-lms-to-diverse-code-edits/) (ICML 2025).
- **Open Source Engineer (NumPy)**, [Quansight Labs](https://labs.quansight.org/) · *Jul 2024 – Present*<br>Author and maintainer of [`numpy-quaddtype`](https://github.com/SwayamInSync/numpy-quaddtype), a cross-platform 128-bit quad-precision dtype (100k+ downloads) built on NumPy's new C DType API.
- **Open Source Research Engineer**, [BigCode](https://www.bigcode-project.org/) · *Feb 2023 – 2024* <!-- 🚧 MARKER: confirm end date -->
<br>Contributed to [StarCoder](https://arxiv.org/abs/2305.06161) (15.5B parameters, 1T tokens) and [OctoPack](https://arxiv.org/abs/2308.07124) for instruction tuning of code models.
- **Machine Learning Engineer Intern**, dataX.ai (CrowdANALYTX) · *May 2022 – Nov 2022*<br>Deep-learning models for vision and language; built an ONNX conversion API and custom CUDA kernels for a 2× segmentation speedup.
- **Data Science Intern**, Scaler (InterviewBit) · *2022* <!-- 🚧 MARKER: confirm dates -->
<br>Built predictive models and data-preprocessing automation, improving user engagement by ~25%.
- **Applied ML Instructor**, Bili Consultancy · *Jan 2022 – Apr 2022*<br>Mentored undergraduate students in applied machine learning.

# 📝 Publications 

<!-- 🚧 MARKER: each paper uses a placeholder figure (images/500x300.png). Drop a real
     teaser image into images/ (e.g. images/nextcoder.png) and update the <img src> below. -->

<div class='paper-box'><div class='paper-box-image'><div><div class="badge">ICML 2025</div><img src='images/500x300.png' alt="NextCoder" width="100%"></div></div>
<div class='paper-box-text' markdown="1">

[NextCoder: Robust Adaptation of Code LMs to Diverse Code Edits](https://www.microsoft.com/en-us/research/publication/nextcoder-robust-adaptation-of-code-lms-to-diverse-code-edits/) \\
Tushar Aggarwal, **Swayam Singh**, Abhijeet Awasthi, Aditya Kanade, Nagarajan Natarajan

[**Paper**](https://www.microsoft.com/en-us/research/publication/nextcoder-robust-adaptation-of-code-lms-to-diverse-code-edits/) \| [**Code** ![](https://img.shields.io/github/stars/microsoft/NextCoder?style=social)](https://github.com/microsoft/NextCoder)

- A synthetic-data generation pipeline and the **SeleKT** adaptation algorithm that make code LLMs robust to diverse, real-world code edits.
- Strong results across **five code-editing benchmarks** while avoiding catastrophic forgetting.
</div>
</div>

<div class='paper-box'><div class='paper-box-image'><div><div class="badge">arXiv 2024</div><img src='images/500x300.png' alt="Narrow Transformer" width="100%"></div></div>
<div class='paper-box-text' markdown="1">

[Narrow Transformer: StarCoder-Based Java-LM For Desktop](https://arxiv.org/abs/2407.03941) \\
**Swayam Singh**, Karthik Rathinasamy, et al. <!-- 🚧 MARKER: confirm full author list -->

[**arXiv**](https://arxiv.org/abs/2407.03941)

- A compact, **Java-specialized** code language model designed to run efficiently on desktop hardware.
</div>
</div>

<div class='paper-box'><div class='paper-box-image'><div><div class="badge">ICLR 2024 Spotlight</div><img src='images/500x300.png' alt="OctoPack" width="100%"></div></div>
<div class='paper-box-text' markdown="1">

[OctoPack: Instruction Tuning Code Large Language Models](https://arxiv.org/abs/2308.07124) \\
Niklas Muennighoff, &hellip;, **Swayam Singh**, et al. <!-- 🚧 MARKER: confirm author ordering -->

[**arXiv**](https://arxiv.org/abs/2308.07124) \| [**Code** ![](https://img.shields.io/github/stars/bigcode-project/octopack?style=social)](https://github.com/bigcode-project/octopack)

- Instruction tuning of code models using natural-language **Git commits** (CommitPack / CommitPackFT).
- Accepted as a **Spotlight (top 5%)** at ICLR 2024.
</div>
</div>

<div class='paper-box'><div class='paper-box-image'><div><div class="badge">TMLR 2023</div><img src='images/500x300.png' alt="StarCoder" width="100%"></div></div>
<div class='paper-box-text' markdown="1">

[StarCoder: May the Source Be With You!](https://arxiv.org/abs/2305.06161) \\
Raymond Li, &hellip;, **Swayam Singh**, et al. (BigCode) <!-- 🚧 MARKER: confirm author ordering -->

[**arXiv**](https://arxiv.org/abs/2305.06161) \| [**Code** ![](https://img.shields.io/github/stars/bigcode-project/starcoder?style=social)](https://github.com/bigcode-project/starcoder)

- A **15.5B-parameter** open code LLM trained on **1T tokens** of permissively licensed code.
- A widely adopted base model for code-generation research.
</div>
</div>

# 🚀 Projects

- [**`numpy-quaddtype`**](https://github.com/SwayamInSync/numpy-quaddtype) — Cross-platform 128-bit quad-precision floating-point dtype for NumPy (100k+ downloads), built on the new NumPy C DType API. *(C · Python)*
- [**QBLAS**](https://github.com/SwayamInSync/QBLAS) — High-performance BLAS for IEEE-754 binary128 (quad) precision. *(C)*
- [**cpp-verify**](https://github.com/SwayamInSync/cpp-verify) — Extending C++ with program-verification constructs backed by SMT solvers, on LLVM. *(C++ · LLVM)*
- [**Clothes Virtual Try-On**](https://github.com/SwayamInSync/clothes-virtual-try-on) — A ViTON-based virtual clothing try-on assistant. *(PyTorch)*
- **MIRA** — Multimodal Image Reconstruction with Attention: transformer-based 2D-to-3D reconstruction. <!-- 🚧 MARKER: add repo/demo link -->

# 🎖 Honors and Awards
- *2024*: **Kaggle Competition Expert** — Bronze medal (top 7%) in UBC-OCEAN; top 3% in the *30 Days of ML* challenge.
- *2024*: **OctoPack** accepted as a **Spotlight (top 5%)** at ICLR 2024.
- *2023*: Selected for the **Amazon ML Summer School 2023**.
- *2023*: **Clothes Virtual Try-On** crossed **275+** GitHub stars.

# 📖 Education
- *2020 – 2024*, **B.Tech**, University of Allahabad, India.<br>Coursework across data structures, algorithms, operating systems, and big data; focus on machine learning with NLP and computer vision.

# 💬 Invited Talks
- *2024.03*: **MAMBA: Zero to Hero** — invited talk on State Space Models at [Cohere for AI](https://cohere.com/research).
<!-- 🚧 MARKER: add more talks here -->
