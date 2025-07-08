# Motif-Calling

**Motif Caller** is a machine learning-based tool that directly detects entire motifs from raw nanopore sequencing signals ("squiggles")—**bypassing the need for traditional basecalling**. This approach enables faster and more accurate decoding of data stored in DNA, especially in systems that encode information using **concatenated motifs from a predefined library**.

This repository contains the full codebase used for preprocessing, model training, and evaluation, as described in our manuscript.

## 🧬 Project Overview

DNA data storage is a promising solution for long-term digital archiving due to DNA's density and durability. However, reading back data typically relies on basecalling — a two-step process that first converts raw nanopore signals into nucleotide sequences and then maps those sequences to stored information.

This is **inefficient and inaccurate** for systems that encode data using motif libraries.

**Motif Caller** addresses this by:
- Learning to **directly predict motifs** from raw nanopore signals
- **Bypassing basecalling entirely**, avoiding loss of signal resolution
- Exploiting rich, motif-level signal features to improve decoding accuracy and efficiency.

## 📁 Repository Structure

```text
Motif-Calling/
├── preprocessing/     # Data preparation
├── training/          # Model training scripts, configs, and dataset loaders
├── evaluation/        # Evaluation metrics
├── prod/              # Production inference scripts for deployment
├── requirements.txt   # Python dependencies
├── environment.yml    # Conda environment
├── README.md          # Project overview (this file)
```

## 🔐 Data Access
The training and test datasets are not included in this repository due to its size. The data is hosted on OSF (https://osf.io/pcdtj/)

## 📜 License
This project is licensed under the MIT License. See LICENSE for full terms.

## Citation
If you use Motif Caller in your research, please cite our manuscript:

Agarwal, Parv, and Thomas Heinis. "Motif Caller: Sequence Reconstruction for Motif-Based DNA Storage." arXiv preprint arXiv:2412.16074 (2024).

## 📬 Contact
Parv Agarwal

Email: parvagrw02@gmail.com

GitHub: @Parvfect
