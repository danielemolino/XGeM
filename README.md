# MedCoDi-M

## Introduction
**MedCoDi-M** (MedicalCoDi-MultiPrompt) is an innovative framework designed to enhance multimodal medical data generation.  
This repository contains the source code, pre-trained models, and usage instructions for **MedCoDi-M**. The goal is to provide an accessible platform for the scientific and clinical community, facilitating the integration of AI models into the diagnostic process.

[![Demo](https://img.shields.io/badge/Demo-View-green)](https://medcodim.unicampus.it/overview)  
[![HuggingFace](https://img.shields.io/badge/HuggingFace-View-blue)](https://huggingface.co/spaces/dmolino/MedCoDi-M)  
[![arXiv](https://img.shields.io/badge/arXiv-1234.56789-b31b1b)](https://arxiv.org/abs/2501.04614)


## Abstract
Artificial Intelligence is revolutionizing medical practice, enhancing diagnostic accuracy and healthcare delivery. However, its adaptation in medical settings still faces significant challenges, related to data availability and privacy constraints. Synthetic data has emerged as a promising solution to mitigate these issues, addressing data scarcity while preserving privacy. Recently, Latent Diffusion Models have emerged as a powerful tool for generating high-quality synthetic data. Meanwhile, the integration of different modalities has gained interest, emphasizing the need of models capable of handle multimodal medical data. Existing approaches struggle to integrate complementary information and lack the ability to generate modalities simultaneously. To address this challenge, we present MedCoDi-M, a 6.77billion-parameter model, designed for multimodal medical data generation, that, following Foundation Model paradigm, exploits contrastive learning and large quantity of data to build a shared latent space which capture the relationships between different data modalities. Further, we introduce the Multi-Prompt training technique, which significantly boosts MedCoDi-M’s generation under different settings. We extensively validate MedCoDi-M: f irst we benchmark it against five competitors on the MIMIC-CXR dataset, a state-of-the-art dataset for Chest X-ray and radiological report generation. Secondly, we perform a Visual Turing Test with expert radiologists to assess the realism and clinical relevance of the generated data, ensuring alignment with real-world scenarios. Finally, we assess the utility of MedCoDi-M in addressing key challenges in the medical field, such as anonymization, data scarcity and imbalance learning. The results are promising, demonstrating the applicability of MedCoDi-M in medical contexts.

![alt text](https://github.com/cosbidev/MedCoDi-M/blob/main/Model.png)

 
## Installation
To install and set up **MedCoDi-M**, follow these steps:

```bash
git clone https://github.com/your-username/MedCoDi-M.git
```
```bash
cd MedCoDi-M
```
```bash
pip install -r requirements.txt
```
## Download the Pretrained Weights
Download the Pretrained weights from [here](https://unicampus365-my.sharepoint.com/:u:/g/personal/daniele_molino_unicampus_it/EaeJqsDx5RNFhAij-80UGdEBuiMw9DrnVqy7cvEstgUo3w?e=xdVpt0) and place it in the Weights folder.

## Demo Instructions
To run the demo, execute the `demo_model.py` script.  
Due to data protection restrictions, real data cannot be shared. Instead, two synthetic images (`Frontal.tiff` and `Lateral.tiff`) are provided in the `Examples` folder.  
The script performs inference on all possible combinations and saves the generated images in `/Examples` folder.


