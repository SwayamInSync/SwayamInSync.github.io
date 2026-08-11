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

I'm a **Research Fellow at [Microsoft Research](https://www.microsoft.com/en-us/research/lab/microsoft-research-india/) India** (AI4Code). I work on **large language models for code**, improving how they reason, follow instructions, and edit real-world codebases, and on developing **efficient algorithms for training LLMs**. More recently, I built the world's first model and pipeline for end-to-end **autoformalization and verification of Rust programs** — a GRPO-based RL recipe with multiplicative gating rewards that lifted the verified-program rate from **1% to 47%** over the SFT baseline, with our **4B Verus-LM outperforming Kimi-K2.5 (1T)** on competitive benchmarks.

The other half of my work is **GPU systems**. I run multi-node distributed training on **Kubernetes-managed NVIDIA B200 clusters**, and I build inference from the metal up: a fused multi-GPU **megakernel** for DeepSeek-V2 (236B) that merges all compute and tensor/expert-parallel communication into a *single* kernel — cross-rank exchange via `multimem` PTX multicast, block-scaled **FP8**, warp specialization, mbarrier synchronization — and a **free-threaded (no-GIL)** re-architecture of vLLM's serving stack that cut inter-engine coordination overhead **~13.3×**. I also write flash-attention kernels for **Blackwell** (tcgen05 2SM MMA) and train MoE models in **NVFP4** 4-bit numerics.

Alongside research, I maintain **open-source** projects across numerical computing and developer tooling. I author and maintain [`numpy-quaddtype`](https://github.com/numpy/numpy-quaddtype), a cross-platform **128-bit (quad-precision) floating-point data type** for NumPy with **100k+ downloads**, as part of [Quansight Labs](https://labs.quansight.org/), and I build [`cpp-verify`](https://github.com/SwayamInSync/cpp-verify), **formal-verification tooling** that extends C++ with **SMT-backed program verification** on top of LLVM. Earlier I contributed to [StarCoder](https://arxiv.org/abs/2305.06161) and [OctoPack](https://arxiv.org/abs/2308.07124) with the [BigCode](https://www.bigcode-project.org/) community, and I'm a **Kaggle Competition Expert**.

My interests sit where machine learning meets systems: making models more capable, and the software they run on faster and more correct. I have <strong><span id='total_cit'>...</span></strong> citations on [Google Scholar](https://scholar.google.com/citations?user=clLJfm8AAAAJ). <!-- the count fills in automatically once the Scholar crawler runs on the deployed repo -->

# 🔥 News

- _2025.05_: &nbsp;🎉 **NextCoder** was accepted at **ICML 2025**. <!-- 🚧 MARKER: verify exact month -->
- _2024.07_: &nbsp;🎉 Joined **Microsoft Research** India as a Research Fellow on the AI4Code team.
- _2024.07_: &nbsp;📦 Released **`numpy-quaddtype`** (quad-precision for NumPy), now with 100k+ downloads.
- _2024.06_: &nbsp;🏅 Reached **Kaggle Competition Expert**.
- _2023.10_: &nbsp;🎉 **OctoPack** accepted as a **Spotlight (top 5%)** at ICLR 2024. <!-- 🚧 MARKER: verify month -->
<!-- 🚧 MARKER: add your latest news items here -->

# 💼 Experience

- **Research Fellow**, [Microsoft Research](https://www.microsoft.com/en-us/research/lab/microsoft-research-india/) India · _Jul 2024 – Present_<br>Training and adapting LLMs for code generation, editing, and reasoning via post-pretraining and online/offline RL — lead author of [NextCoder](https://www.microsoft.com/en-us/research/publication/nextcoder-robust-adaptation-of-code-lms-to-diverse-code-edits/) (ICML 2025), and trained an LLM on par with Gemini and GPT-4o. Built the world's first pipeline for end-to-end autoformalization and verification of Rust programs (**1% → 47%** verified-program rate over the SFT baseline). On the systems side: multi-node distributed training on Kubernetes across NVIDIA **B200** clusters — pod scheduling, GPU resource allocation, distributed-throughput optimization — and a fused multi-GPU inference **megakernel** for DeepSeek-V2 (236B) using `multimem` PTX multicast, block-scaled FP8, and warp specialization.
- **Maintainer**, [NumPy](https://numpy.org/) ([`numpy-quaddtype`](https://github.com/numpy/numpy-quaddtype)) · _Dec 2024 – Present_<br>Author and maintain a cross-platform, true IEEE-compliant 128-bit quad-precision dtype (100k+ downloads) along with efficient BLAS routines. Review and merge core PRs affecting dtype architecture, numerical semantics, and cross-platform behavior, with a focus on numerical correctness and long-term ABI stability.
- **Open Source Intern — Numerical Types & Precision**, [Quansight](https://quansight.com/) · _Jul 2024 – Sep 2024_<br>Led development of [`numpy-quaddtype`](https://github.com/numpy/numpy-quaddtype) on NumPy's new C DType API: casting, ufunc dispatch, scalar types, memory model, and serialization. Built multi-platform distribution and testing infrastructure handling compiler toolchains, SIMD availability, FMA support, and architecture-specific behavior.
- **Open Source Research Engineer**, [BigCode](https://www.bigcode-project.org/) · _Feb 2023 – 2024_ <!-- 🚧 MARKER: confirm end date -->
  <br>Contributed to [StarCoder](https://arxiv.org/abs/2305.06161) (15.5B parameters, 1T tokens) and [OctoPack](https://arxiv.org/abs/2308.07124) for instruction tuning of code models.
- **Machine Learning Engineer Intern**, dataX.ai (CrowdANALYTX) · _May 2022 – Oct 2022_<br>Deployed vision/language models via **NVIDIA Triton Inference Server** (ONNX conversion pipeline), cutting VM load by 12%; wrote custom CUDA kernels for 3D medical-scan processing, achieving a 2× segmentation speedup over existing GPU solutions.
- **Data Science Intern**, Scaler (InterviewBit) · _2022_ <!-- 🚧 MARKER: confirm dates -->
  <br>Built predictive models and data-preprocessing automation, improving user engagement by ~25%.
- **Applied ML Instructor**, Bili Consultancy · _Jan 2022 – Apr 2022_<br>Mentored undergraduate students in applied machine learning.

# 📝 Publications

<div class="pub" markdown="1">
<span class="pub-venue">ICML 2025 · ICLR 2025 DL4Code</span> [**NextCoder: Robust Adaptation of Code LMs to Diverse Code Edits**](https://www.microsoft.com/en-us/research/publication/nextcoder-robust-adaptation-of-code-lms-to-diverse-code-edits/)<br>
**Swayam Singh**, Tushar Aggarwal, et al.

[**Paper**](https://www.microsoft.com/en-us/research/publication/nextcoder-robust-adaptation-of-code-lms-to-diverse-code-edits/) \| [**Code**](https://github.com/microsoft/NextCoder)

- A synthetic-data generation pipeline and **SeleKT** (Selective Knowledge Transfer), a new fine-tuning algorithm that makes code LLMs robust to diverse, real-world code edits.
- Our **32B** model matches **Gemini** and beats **GPT-4o** on harder tasks like **aider-polyglot** — achieved purely via **SFT**, with SeleKT substantially reducing catastrophic forgetting compared to other methods.
</div>

<div class="pub" markdown="1">
<span class="pub-venue">arXiv 2024</span> [**Narrow Transformer: StarCoder-Based Java-LM For Desktop**](https://arxiv.org/abs/2407.03941)<br>
Kamalkumar Rathinasamy, …, **Swayam Singh**, et al.

[**arXiv**](https://arxiv.org/abs/2407.03941)

- A compact, **Java-specialized** code language model designed to run efficiently on desktop hardware.
</div>

<div class="pub" markdown="1">
<span class="pub-venue">ICLR 2024 Spotlight · NeurIPS 2023 Instruction Workshop</span> [**OctoPack: Instruction Tuning Code Large Language Models**](https://arxiv.org/abs/2308.07124)<br>
Niklas Muennighoff, …, **Swayam Singh**, et al.

[**arXiv**](https://arxiv.org/abs/2308.07124) \| [**Code**](https://github.com/bigcode-project/octopack)

- Instruction tuning of code models using natural-language **Git commits** (CommitPack / CommitPackFT).
- Accepted as a **Spotlight (top 5%)** at ICLR 2024.
</div>

<div class="pub" markdown="1">
<span class="pub-venue">TMLR 2023</span> [**StarCoder: May the Source Be With You!**](https://arxiv.org/abs/2305.06161)<br>
Raymond Li, …, **Swayam Singh**, et al. (BigCode)

[**arXiv**](https://arxiv.org/abs/2305.06161) \| [**Code**](https://github.com/bigcode-project/starcoder)

- A **15.5B-parameter** open code LLM trained on **1T tokens** of permissively licensed code.
- A widely adopted base model for code-generation research.
</div>

# 🚀 Projects

- **Free-Threaded vLLM Serving** — Re-architected vLLM's multi-process inference server into a single **free-threaded (no-GIL)** process, replacing ZMQ/msgpack and SHM/pickle IPC between the API layer, scheduler, and per-GPU workers with in-process reference passing, and removing the single-uvloop API-server bottleneck via per-frontend threads. Cut inter-engine coordination overhead **~13.3×** (94.7 ms → 7.1 ms wall-clock union) and P99 inter-token latency from ~540 ms to **~112 ms**; best configuration reached 61.96 req/s vs 59.33 stock. Kept CUDA graphs compatible in the decode path and added a thread-local tokenizer. _(Python no-GIL · CUDA)_
- [**Bare-Bones Inference**](https://github.com/SwayamInSync/bare-bones) — An inference stack with engine-grade batching and scheduling that serves a single fused megakernel instead of a multi-kernel pipeline. Built DeepSeek-V2 (236B)'s block-scaled FP8-quantized, multi-GPU megakernel for NVIDIA B200 (sm_100) GPUs that fuses all computation and TP/EP communication into one kernel via `multimem` PTX multicast writes (directly writing into peer ranks' memory at once); achieves **13.11 ms/token** decode, beating vLLM's optimized serving latency at B=1 by 2× and eager mode by >10×. _(CUDA · C++ · PTX)_
- **Blackwell Flash-Attention Kernels** — Prototyped flash-attention kernels for NVIDIA Blackwell (B200, sm_100) using **tcgen05 2SM MMA** and warp specialization, with mbarrier-based producer/consumer synchronization and a persistent-CTA design; profiled against roofline limits using CuTe layouts. _(CUDA · PTX)_
- **NVFP4 MoE Training System** — Built a Mixture-of-Experts training system from scratch targeting NVIDIA B200 with **NVFP4** (4-bit) numerics, focused on low-precision training stability and multi-GPU throughput. _(CUDA · C++)_
- [**numpy-quaddtype**](https://github.com/numpy/numpy-quaddtype) — Cross-platform 128-bit (quad-precision) floating-point dtype for NumPy (100k+ downloads); a portable alternative to long double with full casting, ufunc dispatch, scalar types, and serialization that behaves consistently across compilers and architectures. _(C, C++, Python)_
- [**QBLAS**](https://github.com/SwayamInSync/QBLAS) — High-performance BLAS for IEEE-754 binary128 (quad) precision, with optimized linear-algebra kernels that bring quad-precision numerics to workloads unsupported by standard double-precision libraries. _(C++)_
- [**cpp-verify**](https://github.com/SwayamInSync/cpp-verify) — Extends C++ with first-class formal-verification constructs (pre/post-conditions and invariants), lowering specifications through an LLVM-based pipeline and discharging the resulting proof obligations to SMT solvers. _(C++ · LLVM · SMT)_
- [**Clothes Virtual Try-On**](https://github.com/SwayamInSync/clothes-virtual-try-on) — An end-to-end virtual try-on system: ResNet101/UNet garment-body segmentation, OpenPose pose estimation, and a PyTorch VITON warping pipeline (SSIM 0.895, 500+ ⭐). _(PyTorch)_
- [**MIRA**](https://github.com/swayamInSync/mira) — Multimodal Image Reconstruction with Attention: transformer-based single-view text/image-to-3D using ViT encoders and triplane decoders with cross-attention; generates 3D mesh and video in under 10s on an A100. _(PyTorch)_

# 🛠 Skills

- **Systems, GPU & Serving** — CUDA kernels and megakernels, multi-GPU / multi-node deployment, tensor/expert parallelism, cross-rank collectives (`multimem` / NVLink), CUDA graphs, FP8 and NVFP4 quantization, inference serving (vLLM internals), NVIDIA Triton Inference Server, SIMD/FMA, performance optimization
- **Infra & Distributed** — Kubernetes (multi-node GPU workloads, scheduling, resource allocation), distributed training, cross-platform C-extension packaging & CI
- **Languages** — C, C++, CUDA, CuTeDSL, Python (incl. free-threaded / no-GIL)
- **ML & Training** — LLM post-training, reinforcement learning (online/offline RL), supervised fine-tuning, code-generation models
- **Tools & Frameworks** — PyTorch, CuPy, NumPy (C-API internals), LLVM, SMT solvers
- **Open Source** — Core NumPy maintenance, cross-platform C-extension packaging, code review

# ✍️ Latest Blogs

<div id="latest-posts" class="blog-list">
  <p class="blog-loading">Loading latest posts… <a href="https://swayaminsync.github.io/swayam-script/" target="_blank" rel="noopener">Visit the blog →</a></p>
</div>

# 🎖 Honors and Awards

- _2024_: **Kaggle Competition Expert** — Bronze medal (top 7%) in UBC-OCEAN; top 3% in the _30 Days of ML_ challenge.
- _2024_: Invited to **Google Research Week** — Google Research's gathering of AI researchers (keynote by Jeff Dean; sessions on differential privacy, responsible AI, and more).
- _2024_: **OctoPack** accepted as a **Spotlight (top 5%)** at ICLR 2024.
- _2023_: Selected for the **Amazon ML Summer School 2023**.
- _2023_: **Clothes Virtual Try-On** crossed **500+** GitHub stars.

# 📖 Education

- _2020 – 2024_, **B.Tech**, University of Allahabad, India.<br>Coursework across data structures, algorithms, operating systems, and big data; focus on machine learning with NLP and computer vision.

# 💬 Invited Talks

- **MAMBA: Zero to Hero** — invited talk on State Space Models at [Cohere for AI](https://cohere.com/research).
- **Provably-Correct Code and Efficient Sparse Training of LLMs** — internal research talk at [Microsoft Research](https://www.microsoft.com/en-us/research/lab/microsoft-research-india/) on LLM-driven autoformalization and verification of Rust programs, and efficient sparse training methods for large language models.
- **Foundations of Machine Learning** — a **GDG On-Campus** session on the ML landscape: core ideas, the tooling ecosystem, and where the field is headed.
- **From Deep Learning to Large Language Models** — a talk on the foundations of modern AI: deep learning, generative models, and LLMs.
<!-- 🚧 MARKER: add more talks here -->
