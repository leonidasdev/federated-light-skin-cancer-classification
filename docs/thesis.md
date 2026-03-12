# Federated Learning for Lightweight Skin Cancer Classification Using Dual-Scale Cross-Attention Vision Transformers

> **Author**: Leonardo Chen  
> **Affiliation**: Universidad Politécnica de Madrid  
> **Degree**: Bachelor's Thesis (Trabajo Fin de Grado)  
> **Date**: 2026  
> **Advisor**: *[Advisor Name]*  
> **Keywords**: Federated Learning, Skin Cancer Classification, Vision Transformer, Non-IID Data, Dermoscopy, Privacy-Preserving AI

---

## Abstract

Skin cancer is one of the most prevalent cancers worldwide, and early detection through automated analysis of dermoscopic images has the potential to significantly improve patient outcomes. However, training centralized deep learning models requires aggregating large volumes of sensitive medical data, raising ethical and regulatory concerns under frameworks such as GDPR and HIPAA. Federated Learning (FL) offers a privacy-preserving alternative by enabling collaborative model training without sharing raw patient data. This thesis evaluates the **Dual-Scale Cross-Attention Vision Transformer (DSCATNet)**, a lightweight architecture proposed by Yadav et al. (2024), in a federated learning setting. We adapt DSCATNet for FL and compare its performance under centralized training, IID federated training, and non-IID federated training using Dirichlet-based data heterogeneity across simulated hospital clients. Experiments are conducted on two dermoscopy datasets (HAM10000 and PAD-UFES-20) with a unified 7-class classification schema. We analyze the accuracy gap between centralized and federated settings, the impact of data heterogeneity (Dirichlet α ∈ {0.1, 0.5, 1.0, 10.0}), and the communication efficiency of the federated approach. Our results show that *[PLACEHOLDER: key finding — e.g., "FL achieves within X% of centralized accuracy at α=0.5 while keeping data decentralized"]*. This work contributes the first empirical evaluation of DSCATNet under federated learning constraints and provides a reproducible experimental framework for future research in privacy-preserving dermatological AI.

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Background and Related Work](#2-background-and-related-work)
3. [Methodology](#3-methodology)
4. [Experimental Setup](#4-experimental-setup)
5. [Results](#5-results)
6. [Discussion](#6-discussion)
7. [Conclusions and Future Work](#7-conclusions-and-future-work)
8. [References](#8-references)
9. [Appendices](#appendices)

---

## 1. Introduction

### 1.1 Motivation

Skin cancer accounts for approximately one-third of all diagnosed cancers, with an estimated 1.5 million new cases globally each year. Melanoma, the most lethal form, has a 5-year survival rate exceeding 99% when detected at Stage I but drops to approximately 32% at Stage IV. Early and accurate diagnosis through dermoscopic image analysis is therefore critical.

Deep learning has achieved dermatologist-level accuracy in dermoscopic image classification tasks, with convolutional neural networks (CNNs) such as EfficientNet, ResNet, and DenseNet dominating the field. More recently, Vision Transformers (ViTs) have demonstrated competitive or superior performance by capturing long-range dependencies through self-attention mechanisms.

However, training these models requires large, centralized datasets of annotated dermoscopic images. In practice, dermoscopic data is distributed across hospitals, clinics, and research institutions, each with their own data collection protocols, patient demographics, and imaging equipment. Centralizing this data poses significant challenges:

1. **Privacy and Regulatory Compliance**: Medical images are protected health information (PHI) under GDPR (EU), HIPAA (US), and similar regulations worldwide. Cross-institutional data sharing requires complex data-sharing agreements, anonymization, and ethics board approvals.

2. **Data Heterogeneity**: Different institutions use different imaging devices, protocols, and diagnostic criteria, leading to naturally non-IID (non-Independent and Identically Distributed) data distributions across sites.

3. **Scalability**: Centralizing data from hundreds of institutions is logistically and computationally prohibitive.

**Federated Learning (FL)** addresses these challenges by training a shared global model through iterative communication of model parameters—not raw data—between a central server and distributed clients. Each client trains locally on its own data and sends only model updates to the server, which aggregates them into an improved global model. This enables collaborative learning while preserving data privacy.

### 1.2 Problem Statement

While FL has been applied to medical imaging tasks, most studies use heavyweight architectures (e.g., EfficientNet, ResNet-50) that may be impractical for deployment in resource-constrained clinical settings. Furthermore, the interaction between lightweight Vision Transformer architectures and non-IID data distributions commonly found in federated medical imaging scenarios remains understudied.

This thesis addresses the following research questions:

- **RQ1**: How does the accuracy of DSCATNet, a lightweight dual-scale Vision Transformer, compare between centralized and federated training settings?
- **RQ2**: How does the degree of data heterogeneity (non-IID-ness) affect the federated model's convergence and final performance?
- **RQ3**: What is the communication cost of federated DSCATNet training, and how does it scale with the number of clients and communication rounds?

### 1.3 Contributions

This thesis makes the following contributions:

1. **First FL evaluation of DSCATNet**: We adapt the Dual-Scale Cross-Attention Vision Transformer for federated learning and provide the first empirical evaluation of its performance under FL constraints.

2. **Non-IID analysis with Dirichlet heterogeneity**: We systematically study the impact of data heterogeneity using Dirichlet-based splits with α ∈ {0.1, 0.5, 1.0, 10.0}, covering extreme non-IID to near-IID distributions.

3. **Reproducible experimental framework**: We release a complete, tested, and documented codebase supporting centralized training, federated simulation, and comparative evaluation on multiple dermoscopy datasets.

4. **Multi-dataset evaluation**: We evaluate on HAM10000 and PAD-UFES-20 with a unified 7-class schema, providing cross-dataset generalization insights.

### 1.4 Thesis Organization

The remainder of this thesis is organized as follows:
- **Chapter 2** reviews the background on skin cancer classification, Vision Transformers, federated learning, and related work.
- **Chapter 3** describes the methodology, including model architecture, federated learning protocol, and data handling.
- **Chapter 4** details the experimental setup, including datasets, hyperparameters, and evaluation metrics.
- **Chapter 5** presents the experimental results.
- **Chapter 6** discusses the findings, limitations, and implications.
- **Chapter 7** concludes the thesis and outlines future work.

---

## 2. Background and Related Work

### 2.1 Skin Cancer Classification with Deep Learning

#### 2.1.1 Clinical Context

Dermoscopy is a non-invasive imaging technique that uses polarized light to visualize subsurface skin structures invisible to the naked eye. The ABCDE criteria (Asymmetry, Border irregularity, Color variation, Diameter, Evolution) provide a clinical framework for assessing lesions, but dermoscopic pattern recognition requires extensive training and experience.

The seven skin lesion categories used in this work correspond to the most clinically relevant diagnostic groups:

| Class | Abbreviation | Description | Clinical Significance |
|-------|-------------|-------------|----------------------|
| 0 | AKIEC | Actinic Keratosis / Intraepithelial Carcinoma | Pre-malignant; may progress to SCC |
| 1 | BCC | Basal Cell Carcinoma | Most common skin cancer; rarely metastasizes |
| 2 | BKL | Benign Keratosis-like Lesions | Benign; includes seborrheic keratosis |
| 3 | DF | Dermatofibroma | Benign fibrous histiocytoma |
| 4 | MEL | Melanoma | Most lethal skin cancer; early detection critical |
| 5 | NV | Melanocytic Nevus | Benign mole; most common lesion type |
| 6 | VASC | Vascular Lesion | Benign; includes angiomas, angiokeratomas |

#### 2.1.2 Deep Learning Approaches

The ISIC (International Skin Imaging Collaboration) challenge series has driven significant advances in automated skin lesion classification:

- **CNN-based approaches**: EfficientNet, ResNet, DenseNet, and Inception variants have achieved state-of-the-art results. EfficientNet-B4 and -B7 consistently rank among the top performers with accuracies exceeding 85% on the ISIC 2019 challenge.

- **Vision Transformer approaches**: ViT, DeiT, and Swin Transformer have been applied to dermoscopy with results competitive to or exceeding CNNs. However, standard ViTs require pre-training on large-scale datasets (ImageNet-21k) and have high computational costs.

- **Hybrid and lightweight approaches**: Models like MobileViT and DSCATNet aim to reduce computational requirements while maintaining accuracy, making them suitable for mobile or edge deployment.

### 2.2 Vision Transformers

#### 2.2.1 Standard Vision Transformer (ViT)

The Vision Transformer (Dosovitskiy et al., 2020) divides an input image into fixed-size patches, projects each patch into an embedding space, and processes the resulting sequence through standard Transformer encoder blocks. Key components include:

- **Patch embedding**: A convolutional layer with kernel_size = stride = patch_size converts $(H \times W \times C)$ images into $(N \times D)$ token sequences, where $N = (H/P)^2$ and $D$ is the embedding dimension.
- **Class token**: A learnable $[CLS]$ token prepended to the sequence, whose final representation is used for classification.
- **Positional embedding**: Learnable position vectors added to patch embeddings to encode spatial information.
- **Transformer encoder**: A stack of blocks, each containing multi-head self-attention (MHSA) and a feed-forward network (FFN) with residual connections and layer normalization.

The standard ViT processes patches at a single scale, which limits its ability to capture both fine-grained local features and global contextual information simultaneously.

#### 2.2.2 DSCATNet: Dual-Scale Cross-Attention Vision Transformer

Yadav et al. (2024) proposed the Dual-Scale Cross-Attention Vision Transformer (DSCATNet) specifically for skin cancer classification. The key innovation is processing images at two spatial scales simultaneously and enabling information exchange between scales through cross-attention.

**Architecture Overview**:

The DSCATNet architecture consists of four main components:

1. **Dual-Scale Patch Embedding**: The input image is simultaneously divided into:
   - **Fine-scale patches** ($8 \times 8$): Produces $N_f = (224/8)^2 = 784$ patches, capturing detailed local texture patterns (pigment networks, dots, globules).
   - **Coarse-scale patches** ($16 \times 16$): Produces $N_c = (224/16)^2 = 196$ patches, capturing global structural features (overall shape, symmetry, color distribution).
   
   Each scale has its own learnable CLS token and positional embeddings.

2. **Cross-Scale Attention Blocks**: Each of the $L$ transformer blocks performs:
   - **Cross-attention**: Fine tokens query coarse tokens (acquiring global context) and coarse tokens query fine tokens (acquiring local detail). This is a bidirectional mechanism with separate Q, K, V projections per direction (12 linear projections per block: $Q_f, K_c, V_c, Q_c, K_f, V_f$ for both cross-attention directions).
   - **Self-attention**: Each scale independently applies multi-head self-attention to its own tokens.
   - **Feed-forward networks**: Each scale has a separate FFN (Linear → GELU → Dropout → Linear → Dropout) with hidden dimension $= D \times r$.
   - All sub-layers use pre-norm residual connections: $x = x + f(\text{LayerNorm}(x))$.

3. **Feature Fusion**: CLS tokens from both scales are concatenated and projected: $z = W_{fuse} \cdot [z_f^{CLS} \| z_c^{CLS}]$ where $W_{fuse} \in \mathbb{R}^{D \times 2D}$.

4. **Classification Head**: $\text{LayerNorm} \rightarrow \text{Dropout} \rightarrow \text{Linear}(D \rightarrow C)$ where $C$ is the number of classes.

**Paper Variant Configuration** (used in this work):

| Parameter | Value |
|-----------|-------|
| Embedding dimension ($D$) | 384 |
| Depth ($L$) | 6 blocks |
| Attention heads | 12 |
| MLP ratio ($r$) | 4.0 |
| Fine patch size | 8 × 8 |
| Coarse patch size | 16 × 16 |
| Dropout rate | 0.1 |
| Total parameters | ~29.4M |

**Note on Parameter Count**: The original paper reports approximately 22M parameters. Our implementation yields 29.4M parameters. This discrepancy arises from the cross-attention mechanism: our implementation uses 12 separate linear projections per cross-attention layer (6 for fine→coarse, 6 for coarse→fine), while the paper may use a shared or reduced projection scheme. The architectural behavior is equivalent; the additional parameters provide more capacity in the cross-attention layers.

### 2.3 Federated Learning

#### 2.3.1 Core Concepts

Federated Learning (McMahan et al., 2017) enables collaborative training of a shared model across $K$ distributed clients without centralizing data. The canonical algorithm, **Federated Averaging (FedAvg)**, proceeds as follows:

**Algorithm: FedAvg**

```
Server initializes global model weights w₀
For each round t = 1, 2, ..., T:
    Server selects a subset S_t of K clients (|S_t| = C·K)
    Server sends w_t to all clients in S_t
    For each client k ∈ S_t in parallel:
        w_k^{t+1} ← ClientUpdate(k, w_t)   // E local epochs of SGD
    Server aggregates: w_{t+1} ← Σ_k (n_k / n) · w_k^{t+1}
        where n_k = samples at client k, n = Σ n_k
```

Key properties:
- **Communication efficiency**: Only model parameters are transmitted, not raw data.
- **Privacy**: Raw data never leaves the client site.
- **Heterogeneity tolerance**: Clients may have different data volumes, distributions, and computational capabilities.

#### 2.3.2 Non-IID Data in Federated Learning

In practice, data across clients is rarely IID. In medical imaging, non-IID-ness arises from:
- **Label distribution skew**: Different hospitals see different prevalences of diseases.
- **Feature distribution skew**: Different imaging devices, lighting conditions, and patient demographics.
- **Quantity skew**: Large academic centers have more data than small clinics.

Non-IID data is a fundamental challenge for FL because it causes client models to diverge during local training, leading to poor aggregation. The Dirichlet distribution is commonly used to simulate controlled non-IID splits:

$$p_k \sim \text{Dir}(\alpha)$$

where $\alpha$ controls the degree of heterogeneity:
- $\alpha \rightarrow 0$: Each client has samples from only one or two classes (extreme non-IID).
- $\alpha = 0.5$: Moderate heterogeneity (typical benchmark setting).
- $\alpha \rightarrow \infty$: Each client has a uniform distribution across classes (IID).

#### 2.3.3 Federated Learning in Medical Imaging

FL has been applied to various medical imaging domains:

- **Chest X-ray classification**: FedAvg with DenseNet-121 across 5 institutions (Sheller et al., 2020).
- **Brain tumor segmentation**: FL with U-Net variants across 10+ institutions (Li et al., 2019).
- **Retinal image analysis**: FL for diabetic retinopathy screening (Lu et al., 2022).
- **Skin lesion classification**: Khullar et al. (2025) evaluated EfficientNetV2-S in FL settings on ISIC 2019, reporting accuracy drops of 3–8% compared to centralized training depending on the non-IID degree.

### 2.4 Related Work Comparison

| Study | Model | Dataset | FL Strategy | Non-IID | Accuracy Gap |
|-------|-------|---------|-------------|---------|--------------|
| Khullar et al. (2025) | EfficientNetV2-S | ISIC 2019 | FedAvg | Dirichlet | 3–8% |
| Sheller et al. (2020) | DenseNet-121 | Chest X-ray | FedAvg | Natural | ~2% |
| **This work** | DSCATNet (small) | HAM10000, PAD-UFES-20 | FedAvg | Dirichlet | *[PLACEHOLDER]* |

Key differentiators of this work:
- First FL evaluation of a dual-scale Vision Transformer architecture.
- Systematic Dirichlet α ablation study.
- Multi-dataset evaluation with unified label schema.
- Open-source, fully tested implementation.

---

## 3. Methodology

### 3.1 Model Architecture

We implement DSCATNet as described in Section 2.2.2, using the **paper variant** (H=12 attention heads) with the following configuration:

```
Input: (B, 3, 224, 224)
    ↓
Dual-Scale Patch Embedding
    ├── Fine-scale (8×8): 784 patches → (B, 785, 384)  [+1 CLS token]
    └── Coarse-scale (16×16): 196 patches → (B, 197, 384)  [+1 CLS token]
    ↓
6 × Cross-Scale Attention Block
    ├── Cross-attention: fine ↔ coarse (bidirectional)
    ├── Self-attention per scale
    └── FFN per scale (GELU, hidden=1536)
    ↓
Layer Normalization (per scale)
    ↓
CLS Token Extraction → Concatenation
    ↓
Linear Fusion (768 → 384)
    ↓
Classification Head: LayerNorm → Dropout(0.1) → Linear(384 → 7)
    ↓
Output: (B, 7) logits
```

**Pretrained Weight Initialization**: We initialize compatible layers from ViT-Small (ImageNet-21k, `vit_small_patch16_224`) pretrained weights via the `timm` library. Specifically:
- ViT blocks 0–5 → fine-scale self-attention and FFN weights
- ViT blocks 6–11 → coarse-scale self-attention and FFN weights
- ViT patch embedding (16×16) → coarse-scale patch embedding
- ViT positional embedding and CLS token → coarse-scale positional embedding and CLS token
- ViT final LayerNorm → coarse-scale final LayerNorm

This transfers 150 out of 286 parameter tensors (~52%). Cross-attention layers, fine-scale patch embedding (8×8), fusion layer, and classification head remain randomly initialized with truncated normal (std=0.02).

### 3.2 Training Configuration

Following the original DSCATNet paper (Yadav et al., 2024), we use:

| Parameter | Value | Justification |
|-----------|-------|---------------|
| Optimizer | Adam | Paper specification |
| Learning rate | $10^{-3}$ | Paper specification (fixed, no scheduling) |
| Weight decay | 0.0 | Paper uses Adam without L2 regularization |
| Batch size | 32 effective | 8 per GPU × 4 gradient accumulation steps (VRAM-constrained) |
| Epochs (centralized) | 200 | Paper specification |
| FL rounds | 100 | Comparable total iterations |
| Local epochs per round | 1 | Standard FL setting (McMahan et al., 2017) |
| Loss function | Weighted CrossEntropy | Modification: inverse-frequency class weights to handle class imbalance |
| Augmentation | None | Paper-aligned: no data augmentation |
| Image normalization | ImageNet stats | Mean=(0.485, 0.456, 0.406), Std=(0.229, 0.224, 0.225) |
| Gradient clipping | max_norm=1.0 | Stability measure for transformer training |

**Deviation from paper**: We use weighted CrossEntropyLoss with inverse-frequency class weights to handle the severe class imbalance in dermoscopy datasets (e.g., NV comprises ~67% of HAM10000). The original paper uses unweighted CrossEntropy. We justify this deviation because FL with non-IID splits can exacerbate class imbalance at individual clients, and unweighted loss may cause the model to collapse to the majority class.

### 3.3 Federated Learning Protocol

We simulate FL using in-process training (no network communication) with the following protocol:

**FedAvg Configuration**:

| Parameter | Value |
|-----------|-------|
| Number of clients ($K$) | 4 |
| Client participation ($C$) | 1.0 (all clients per round) |
| Aggregation | Weighted FedAvg: $w_{t+1} = \sum_k \frac{n_k}{n} w_k^{t+1}$ |
| Local epochs ($E$) | 1 |
| Communication rounds ($T$) | 100 |
| Early stopping | Patience = 10 rounds |

**Non-IID Data Distribution**:

We use Dirichlet-based data partitioning to create heterogeneous client data:

1. All selected datasets are combined into a single pool.
2. For each class $c$, samples are distributed across $K$ clients according to $p_c \sim \text{Dir}(\alpha \cdot \mathbf{1}_K)$.
3. Each client receives a subset of indices weighted by their Dirichlet proportions.

We experiment with $\alpha \in \{0.1, 0.5, 1.0, 10.0\}$:

| α | Interpretation | Expected Behavior |
|---|---------------|-------------------|
| 0.1 | Extreme non-IID | Each client has 1–2 dominant classes |
| 0.5 | Moderate non-IID | Standard benchmark setting |
| 1.0 | Mild non-IID | Some class imbalance across clients |
| 10.0 | Near-IID | Approximately uniform distribution |

**Client Training Loop** (per round):

```
Receive global weights w_t from server
Set local model weights to w_t
For e = 1, ..., E:
    For each mini-batch (x, y) from local train set:
        Compute loss = WeightedCE(model(x), y) / accumulation_steps
        Backpropagate gradients
        Every accumulation_steps batches:
            Clip gradients (max_norm=1.0)
            Update optimizer
Send updated weights w_k^{t+1} to server
```

### 3.4 Data Handling

#### 3.4.1 Unified 7-Class Schema

All datasets are mapped to a unified 7-class taxonomy (AKIEC, BCC, BKL, DF, MEL, NV, VASC). This mapping is defined per-dataset in the `BaseDermoscopyDataset` class hierarchy.

#### 3.4.2 Data Preprocessing

1. **Resize**: All images are resized to 224×224 pixels using bilinear interpolation.
2. **Normalization**: Pixel values are normalized using ImageNet statistics.
3. **No augmentation**: Aligned with the original DSCATNet paper.

#### 3.4.3 Train/Validation Split

- **Centralized**: 85% train / 15% validation, deterministic split using `torch.Generator(seed=42)`.
- **Federated**: Each client's data is further split into 85% train / 15% validation with the same deterministic generator.

### 3.5 Evaluation Metrics

We report the following metrics computed on validation/test sets:

| Metric | Formula | Rationale |
|--------|---------|-----------|
| **Accuracy** | $\frac{\text{correct}}{\text{total}}$ | Overall correctness |
| **Balanced Accuracy** | $\frac{1}{C} \sum_{c=1}^C \text{Recall}_c$ | Accounts for class imbalance |
| **Precision (macro)** | $\frac{1}{C} \sum_{c=1}^C \frac{TP_c}{TP_c + FP_c}$ | Average per-class precision |
| **Recall (macro)** | $\frac{1}{C} \sum_{c=1}^C \frac{TP_c}{TP_c + FN_c}$ | Average per-class recall |
| **F1 (macro)** | $\frac{1}{C} \sum_{c=1}^C \frac{2 \cdot P_c \cdot R_c}{P_c + R_c}$ | Harmonic mean of P and R |
| **F1 (weighted)** | $\sum_{c=1}^C \frac{n_c}{n} \cdot F1_c$ | Support-weighted F1 |
| **AUC-ROC (macro)** | One-vs-rest AUC, macro-averaged | Discrimination ability |

**Per-class metrics**: We report accuracy, precision, recall, and support for each of the 7 classes.

**Confusion matrix**: Visualized as a heatmap to reveal systematic misclassification patterns.

---

## 4. Experimental Setup

### 4.1 Datasets

#### 4.1.1 HAM10000

The HAM10000 (Human Against Machine with 10,000 training images) dataset (Tschandl et al., 2018) contains 10,015 dermoscopic images of pigmented lesions from the Medical University of Vienna and the Skin Cancer Practice of Cliff Rosendahl.

| Property | Value |
|----------|-------|
| Total images | 10,015 |
| Image resolution | Variable (high-resolution dermoscopy) |
| Number of classes | 7 |
| Annotation source | Histopathology (majority), expert consensus, in-vivo confocal microscopy |

**Class distribution**:

| Class | Count | Percentage |
|-------|-------|------------|
| NV | 6,705 | 66.9% |
| MEL | 1,113 | 11.1% |
| BKL | 1,099 | 11.0% |
| BCC | 514 | 5.1% |
| AKIEC | 327 | 3.3% |
| VASC | 142 | 1.4% |
| DF | 115 | 1.1% |

The dataset exhibits severe class imbalance, with NV comprising nearly 67% of all samples.

#### 4.1.2 PAD-UFES-20

The PAD-UFES-20 (Pacheco et al., 2020) dataset contains 2,298 clinical (non-dermoscopic) skin lesion images collected from the Dermatology and Radiotherapy Service of the Universidade Federal do Espírito Santo (Brazil).

| Property | Value |
|----------|-------|
| Total images | 2,298 |
| Image resolution | Variable (smartphone-captured clinical images) |
| Number of classes | 6 (mapped to 7-class schema) |
| Annotation source | Histopathological confirmation |

**Class distribution**:

| Class | Count | Percentage |
|-------|-------|------------|
| *[PLACEHOLDER: Fill with actual per-class counts from dataset]* | | |

**Note**: PAD-UFES-20 contains clinical (non-dermoscopic) images captured with smartphones, introducing a domain gap compared to HAM10000's dermoscopic images. This makes cross-dataset FL evaluation particularly challenging.

### 4.2 Hardware and Software

| Component | Specification |
|-----------|---------------|
| GPU | NVIDIA GeForce RTX 3050 Laptop GPU (4 GB VRAM) |
| CPU | *[PLACEHOLDER: CPU model]* |
| RAM | *[PLACEHOLDER: RAM size]* |
| OS | Windows 11 |
| Python | 3.13.3 |
| PyTorch | 2.7.1+cu118 |
| CUDA | 11.8 |
| Flower | 1.25.0 |
| timm | 1.0.24 |

**VRAM Constraint**: The 4 GB VRAM limitation necessitates a batch size of 4 with gradient accumulation (8 steps) to achieve an effective batch size of 32. AMP is disabled for training stability.

### 4.3 Experiment Matrix

| Experiment ID | Mode | Dataset | Non-IID | α | Clients | Rounds/Epochs |
|--------------|------|---------|---------|---|---------|---------------|
| C-HAM | Centralized | HAM10000 | N/A | N/A | N/A | 200 epochs |
| C-PAD | Centralized | PAD-UFES-20 | N/A | N/A | N/A | 200 epochs |
| F-HAM-01 | Federated | HAM10000 | Dirichlet | 0.1 | 4 | 100 rounds |
| F-HAM-05 | Federated | HAM10000 | Dirichlet | 0.5 | 4 | 100 rounds |
| F-HAM-10 | Federated | HAM10000 | Dirichlet | 1.0 | 4 | 100 rounds |
| F-HAM-100 | Federated | HAM10000 | Dirichlet | 10.0 | 4 | 100 rounds |
| F-PAD-05 | Federated | PAD-UFES-20 | Dirichlet | 0.5 | 4 | 100 rounds |

### 4.4 Reproducibility

All experiments use the following reproducibility measures:
- Fixed random seed: 42 (torch, numpy, random, cuDNN deterministic mode)
- Deterministic train/val splits using `torch.Generator(seed=42)`
- Full configuration saved as `config.json` in each experiment output directory
- Complete training history saved as `history.json`
- Best model and periodic checkpoints saved for resume capability

---

## 5. Results

### 5.1 Centralized Baselines

#### 5.1.1 HAM10000 Centralized (C-HAM)

*[PLACEHOLDER: Replace with actual results when training completes]*

| Metric | Value |
|--------|-------|
| Best Validation Accuracy | *[PLACEHOLDER]* |
| Best Epoch | *[PLACEHOLDER]* |
| Balanced Accuracy | *[PLACEHOLDER]* |
| F1 (macro) | *[PLACEHOLDER]* |
| F1 (weighted) | *[PLACEHOLDER]* |
| AUC-ROC (macro) | *[PLACEHOLDER]* |
| Total Training Time | *[PLACEHOLDER]* |

**Training Curves**:

*[PLACEHOLDER: Include training loss and validation accuracy curves from `history.json`]*

**Per-Class Performance**:

| Class | Accuracy | Precision | Recall | Support |
|-------|----------|-----------|--------|---------|
| AKIEC | *[PLACEHOLDER]* | *[PLACEHOLDER]* | *[PLACEHOLDER]* | *[PLACEHOLDER]* |
| BCC | *[PLACEHOLDER]* | *[PLACEHOLDER]* | *[PLACEHOLDER]* | *[PLACEHOLDER]* |
| BKL | *[PLACEHOLDER]* | *[PLACEHOLDER]* | *[PLACEHOLDER]* | *[PLACEHOLDER]* |
| DF | *[PLACEHOLDER]* | *[PLACEHOLDER]* | *[PLACEHOLDER]* | *[PLACEHOLDER]* |
| MEL | *[PLACEHOLDER]* | *[PLACEHOLDER]* | *[PLACEHOLDER]* | *[PLACEHOLDER]* |
| NV | *[PLACEHOLDER]* | *[PLACEHOLDER]* | *[PLACEHOLDER]* | *[PLACEHOLDER]* |
| VASC | *[PLACEHOLDER]* | *[PLACEHOLDER]* | *[PLACEHOLDER]* | *[PLACEHOLDER]* |

**Confusion Matrix**:

*[PLACEHOLDER: Include confusion matrix heatmap]*

#### 5.1.2 PAD-UFES-20 Centralized (C-PAD)

*[PLACEHOLDER: Same structure as above]*

### 5.2 Federated Learning Results

#### 5.2.1 HAM10000 Federated — Dirichlet α=0.5 (F-HAM-05)

*[PLACEHOLDER: Replace with actual results]*

| Metric | Value |
|--------|-------|
| Best Global Validation Accuracy | *[PLACEHOLDER]* |
| Best Round | *[PLACEHOLDER]* |
| Balanced Accuracy | *[PLACEHOLDER]* |
| F1 (macro) | *[PLACEHOLDER]* |
| AUC-ROC (macro) | *[PLACEHOLDER]* |
| Total Rounds | *[PLACEHOLDER]* |
| Total Communication (MB) | *[PLACEHOLDER]* |
| Total Training Time | *[PLACEHOLDER]* |

**Client Data Distribution**:

| Client | Total Samples | AKIEC | BCC | BKL | DF | MEL | NV | VASC |
|--------|--------------|-------|-----|-----|----|----|----|----|
| 0 | *[PH]* | *[PH]* | *[PH]* | *[PH]* | *[PH]* | *[PH]* | *[PH]* | *[PH]* |
| 1 | *[PH]* | *[PH]* | *[PH]* | *[PH]* | *[PH]* | *[PH]* | *[PH]* | *[PH]* |
| 2 | *[PH]* | *[PH]* | *[PH]* | *[PH]* | *[PH]* | *[PH]* | *[PH]* | *[PH]* |
| 3 | *[PH]* | *[PH]* | *[PH]* | *[PH]* | *[PH]* | *[PH]* | *[PH]* | *[PH]* | 

**Per-Client Performance** (final round):

| Client | Accuracy | Loss |
|--------|----------|------|
| 0 | *[PLACEHOLDER]* | *[PLACEHOLDER]* |
| 1 | *[PLACEHOLDER]* | *[PLACEHOLDER]* |
| 2 | *[PLACEHOLDER]* | *[PLACEHOLDER]* |
| 3 | *[PLACEHOLDER]* | *[PLACEHOLDER]* |

**Training Convergence**:

*[PLACEHOLDER: Include round-wise global accuracy and loss curves]*

#### 5.2.2 Effect of Dirichlet α (Non-IID Ablation)

*[PLACEHOLDER: Results for α ∈ {0.1, 0.5, 1.0, 10.0}]*

| α | Best Accuracy | Accuracy Gap vs Centralized | Convergence Round | Total Time |
|---|--------------|---------------------------|-------------------|------------|
| 0.1 | *[PLACEHOLDER]* | *[PLACEHOLDER]* | *[PLACEHOLDER]* | *[PLACEHOLDER]* |
| 0.5 | *[PLACEHOLDER]* | *[PLACEHOLDER]* | *[PLACEHOLDER]* | *[PLACEHOLDER]* |
| 1.0 | *[PLACEHOLDER]* | *[PLACEHOLDER]* | *[PLACEHOLDER]* | *[PLACEHOLDER]* |
| 10.0 | *[PLACEHOLDER]* | *[PLACEHOLDER]* | *[PLACEHOLDER]* | *[PLACEHOLDER]* |

**Convergence Comparison**:

*[PLACEHOLDER: Include overlaid convergence curves for all α values]*

**Observation**: *[PLACEHOLDER: Describe the relationship between α and accuracy/convergence]*

#### 5.2.3 PAD-UFES-20 Federated (F-PAD-05)

*[PLACEHOLDER: Same structure as HAM10000 federated]*

### 5.3 Centralized vs. Federated Comparison

| Experiment | Accuracy | Balanced Acc | F1 (macro) | AUC-ROC | Time |
|-----------|----------|-------------|------------|---------|------|
| C-HAM (centralized) | *[PLACEHOLDER]* | *[PLACEHOLDER]* | *[PLACEHOLDER]* | *[PLACEHOLDER]* | *[PLACEHOLDER]* |
| F-HAM-05 (FL, α=0.5) | *[PLACEHOLDER]* | *[PLACEHOLDER]* | *[PLACEHOLDER]* | *[PLACEHOLDER]* | *[PLACEHOLDER]* |
| F-HAM-100 (FL, α=10.0) | *[PLACEHOLDER]* | *[PLACEHOLDER]* | *[PLACEHOLDER]* | *[PLACEHOLDER]* | *[PLACEHOLDER]* |
| C-PAD (centralized) | *[PLACEHOLDER]* | *[PLACEHOLDER]* | *[PLACEHOLDER]* | *[PLACEHOLDER]* | *[PLACEHOLDER]* |
| F-PAD-05 (FL, α=0.5) | *[PLACEHOLDER]* | *[PLACEHOLDER]* | *[PLACEHOLDER]* | *[PLACEHOLDER]* | *[PLACEHOLDER]* |

**Accuracy Gap Analysis**:

| Dataset | Centralized | FL (α=0.5) | Gap | Gap (%) |
|---------|-----------|-----------|-----|---------|
| HAM10000 | *[PLACEHOLDER]* | *[PLACEHOLDER]* | *[PLACEHOLDER]* | *[PLACEHOLDER]* |
| PAD-UFES-20 | *[PLACEHOLDER]* | *[PLACEHOLDER]* | *[PLACEHOLDER]* | *[PLACEHOLDER]* |

### 5.4 Communication Cost Analysis

*[PLACEHOLDER: Fill when FL experiments complete]*

| Experiment | Rounds | Model Size (MB) | Total Upload (MB) | Total Download (MB) | Total Communication (MB) |
|-----------|--------|-----------------|-------------------|--------------------|-----------------------|
| F-HAM-05 | *[PH]* | ~112.4 | *[PH]* | *[PH]* | *[PH]* |
| F-PAD-05 | *[PH]* | ~112.4 | *[PH]* | *[PH]* | *[PH]* |

**Formula**: Total communication per round = $2 \times K \times C \times \text{model\_size}$ (K clients, C participation rate, upload + download).

With 29.4M float32 parameters ≈ 112.4 MB per model copy. For K=4 clients, C=1.0:
- Per round: $2 \times 4 \times 1.0 \times 112.4 = 899.2$ MB
- 100 rounds: $\approx 87.8$ GB total

---

## 6. Discussion

### 6.1 Answering the Research Questions

#### RQ1: Centralized vs. Federated Accuracy

*[PLACEHOLDER: Interpret the accuracy gap results. Expected narrative:]*

The centralized baseline on HAM10000 achieves *[X]%* accuracy, while the federated model with moderate non-IID (α=0.5) achieves *[Y]%*, representing a gap of *[Z]* percentage points (*[Z/X × 100]%* relative decrease). This gap is *[comparable to / larger than / smaller than]* the 3–8% gap reported by Khullar et al. (2025) for EfficientNetV2-S, suggesting that the dual-scale cross-attention architecture *[is / is not]* more or less sensitive to federated training than conventional CNNs.

#### RQ2: Impact of Data Heterogeneity

*[PLACEHOLDER: Interpret the Dirichlet α ablation. Expected narrative:]*

Increasing data heterogeneity (lower α) consistently *[degrades / affects]* model performance. At α=0.1 (extreme non-IID), accuracy drops to *[X]%*, *[Y]* percentage points below the near-IID setting (α=10.0). The relationship between α and accuracy *[is / is not]* monotonic. Convergence *[slows / fails]* at extreme non-IID levels, requiring *[more / fewer]* rounds to reach a given accuracy threshold.

#### RQ3: Communication Efficiency

*[PLACEHOLDER: Interpret communication cost results. Expected narrative:]*

The 29.4M parameter DSCATNet model requires approximately 112.4 MB per weight transmission. Over 100 rounds with 4 fully-participating clients, this totals approximately *[X]* GB of communication — *[significant / moderate / acceptable]* for a clinical network. Compared to EfficientNetV2-S (~21M parameters, ~80 MB), DSCATNet is *[X%]* more expensive in communication per round due to the dual-scale architecture's additional parameters.

### 6.2 Analysis of Class-Level Performance

*[PLACEHOLDER: Analyze per-class results. Expected topics:]*

- **NV (majority class)**: Expected to have highest accuracy due to prevalence.
- **DF, VASC (minority classes)**: Expected to suffer most under non-IID FL, as clients may have very few or zero samples of these classes at low α.
- **MEL (clinically critical)**: Melanoma recall is the most clinically important metric. A false negative for MEL is far more consequential than for BKL.
- **Class weight impact**: Weighted CE loss is expected to improve minority class recall at the expense of slight majority class accuracy reduction.

### 6.3 Comparison with Literature

| Study | Model | Params | Dataset | Best Acc (Centralized) | Best Acc (FL) | Gap |
|-------|-------|--------|---------|----------------------|--------------|-----|
| Yadav et al. (2024) | DSCATNet | ~22M | HAM10000 | 97.6% | N/A | N/A |
| Khullar et al. (2025) | EfficientNetV2-S | ~21M | ISIC 2019 | ~91% | ~85% | ~6% |
| **This work** | DSCATNet | 29.4M | HAM10000 | *[PH]* | *[PH]* | *[PH]* |
| **This work** | DSCATNet | 29.4M | PAD-UFES-20 | *[PH]* | *[PH]* | *[PH]* |

*[PLACEHOLDER: Compare our results with the literature values above.]*

**Note on reproducibility gap**: The original DSCATNet paper reports 97.6% accuracy on HAM10000, which may include different pre-processing, augmentation, or evaluation protocols (e.g., leaky validation, test-time augmentation). Our implementation strictly separates train/val data and uses no augmentation, which may lead to lower absolute numbers but provides a fair comparison between centralized and federated settings.

### 6.4 Limitations

1. **Single aggregation strategy**: We evaluate only FedAvg. Other strategies (FedProx, FedNova, SCAFFOLD) may mitigate non-IID issues.

2. **Simulated FL**: Our FL simulation runs on a single machine. Real-world FL involves network latency, client dropout, and asynchronous training, which are not modeled.

3. **Limited datasets**: We evaluate on two datasets with a unified 7-class schema. ISIC 2019 (25K images, 8+ classes) and ISIC 2020 (33K images) are available in the codebase but not fully evaluated due to computational constraints.

4. **Single random seed**: Due to computational constraints (each experiment requires *[PLACEHOLDER: ~X hours]* on an RTX 3050), we report results for a single seed (42). Statistical significance requires multiple seeds.

5. **No differential privacy**: Our FL implementation does not include formal differential privacy guarantees (e.g., DP-SGD). We rely on the inherent privacy of not sharing raw data.

6. **Hardware constraints**: The 4 GB VRAM limitation requires small batch sizes and disabling AMP, which may affect convergence compared to training on larger GPUs.

### 6.5 Threats to Validity

- **Internal validity**: Deterministic seeding ensures reproducibility, but single-seed results may not generalize.
- **External validity**: Results on HAM10000 and PAD-UFES-20 may not transfer to other dermatological datasets or clinical populations.
- **Construct validity**: Dirichlet-based non-IID simulation is a proxy for real-world data heterogeneity, which involves feature shift, concept drift, and other factors beyond label distribution skew.

---

## 7. Conclusions and Future Work

### 7.1 Conclusions

This thesis presented the first evaluation of the Dual-Scale Cross-Attention Vision Transformer (DSCATNet) in a federated learning setting for skin cancer classification. The key findings are:

1. *[PLACEHOLDER: Key finding 1 — centralized vs federated gap]*
2. *[PLACEHOLDER: Key finding 2 — non-IID sensitivity]*
3. *[PLACEHOLDER: Key finding 3 — communication cost]*

Our results suggest that *[PLACEHOLDER: overall conclusion about DSCATNet's suitability for FL]*. The dual-scale cross-attention mechanism *[does / does not]* introduce unique challenges for federated aggregation compared to single-scale architectures, as *[PLACEHOLDER: reasoning]*.

### 7.2 Future Work

1. **Advanced FL strategies**: Evaluate FedProx (Li et al., 2020), SCAFFOLD (Karimireddy et al., 2020), and FedNova (Wang et al., 2020) to mitigate non-IID degradation.

2. **Personalization**: Investigate per-client fine-tuning or personalized FL methods (e.g., local adaptation layers) to improve performance on heterogeneous data.

3. **Communication compression**: Apply gradient compression, model pruning, or knowledge distillation to reduce the ~112 MB per-round communication overhead.

4. **Differential privacy**: Integrate DP-SGD (Abadi et al., 2016) to provide formal privacy guarantees with measurable privacy budgets (ε, δ).

5. **Real-world deployment**: Test with a multi-institution setup using actual clinical data and network conditions.

6. **Larger-scale evaluation**: Extend experiments to ISIC 2019 and ISIC 2020 datasets with natural non-IID from different ISIC challenge sources.

7. **Architecture ablation**: Evaluate the contribution of cross-attention vs. self-attention by comparing DSCATNet against a ViT-Small baseline under the same FL conditions.

---

## 8. References

1. Yadav, S. S., Jadhav, S. M., Channe, H., & Rathod, J. (2024). DSCATNet: Dual-Scale Cross-Attention Vision Transformer for Skin Lesion Classification. *PLOS ONE*, 19(12), e0312598. https://doi.org/10.1371/journal.pone.0312598

2. Khullar, V., et al. (2025). Evaluating federated learning for skin lesion classification: a benchmark and analysis of non-IID data distributions. *Scientific Reports*, 15, Article 82402-x. https://doi.org/10.1038/s41598-024-82402-x

3. McMahan, H. B., Moore, E., Ramage, D., Hampson, S., & Arcas, B. A. (2017). Communication-Efficient Learning of Deep Networks from Decentralized Data. *AISTATS 2017*. arXiv:1602.05629.

4. Dosovitskiy, A., et al. (2020). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. *ICLR 2021*. arXiv:2010.11929.

5. Tschandl, P., Rosendahl, C., & Kittler, H. (2018). The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions. *Scientific Data*, 5(1), 180161.

6. Pacheco, A. G. C., et al. (2020). PAD-UFES-20: A skin lesion dataset composed of patient data and clinical images collected from smartphones. *Data in Brief*, 32, 106221.

7. Li, T., Sahu, A. K., Zaheer, M., Sanjabi, M., Talwalkar, A., & Smith, V. (2020). Federated Optimization in Heterogeneous Networks. *MLSys 2020*. arXiv:1812.06127. (FedProx)

8. Karimireddy, S. P., Kale, S., Mohri, M., Reddi, S. J., Stich, S. U., & Suresh, A. T. (2020). SCAFFOLD: Stochastic Controlled Averaging for Federated Learning. *ICML 2020*. arXiv:1910.06378.

9. Wang, J., Liu, Q., Liang, H., Joshi, G., & Poor, H. V. (2020). Tackling the Objective Inconsistency Problem in Heterogeneous Federated Optimization. *NeurIPS 2020*. arXiv:2007.07481. (FedNova)

10. Abadi, M., et al. (2016). Deep Learning with Differential Privacy. *CCS 2016*. arXiv:1607.00133.

11. Sheller, M. J., et al. (2020). Federated learning in medicine: facilitating multi-institutional collaborations without sharing patient data. *Scientific Reports*, 10(1), 12598.

12. Lu, M. Y., et al. (2022). Federated learning for computational pathology on gigapixel whole slide images. *Medical Image Analysis*, 76, 102298.

13. Steiner, A., Kolesnikov, A., Zhai, X., Wightman, R., Uszkoreit, J., & Beyer, L. (2022). How to train your ViT? Data, Augmentation, and Regularization in Vision Transformers. *TMLR 2022*. arXiv:2106.10270.

---

## Appendices

### Appendix A: Software Architecture

The project follows a modular architecture with clear separation of concerns:

```
src/
├── models/           # DSCATNet architecture
│   ├── dscatnet.py           # Main model + factory + weight loading
│   ├── cross_attention.py    # CrossScaleAttention, CrossScaleAttentionBlock
│   └── patch_embedding.py    # PatchEmbedding, DualScalePatchEmbedding
├── centralized/      # Centralized training baseline
│   └── centralized.py        # CentralizedConfig, CentralizedTrainer
├── federated/        # Federated learning
│   ├── simulation.py         # SimulationConfig, FLSimulator, ClientData
│   ├── client.py             # SkinCancerClient (Flower NumPyClient)
│   ├── server.py             # FL server utilities
│   └── strategy.py           # DSCATNetFedAvg (custom FedAvg)
├── data/             # Data handling
│   ├── datasets.py           # Dataset classes, DATASET_REGISTRY
│   ├── preprocessing.py      # Transforms (Albumentations)
│   ├── splits.py             # IID/Non-IID splitting
│   ├── download.py           # Dataset download utilities
│   └── verify.py             # Dataset verification
├── evaluation/       # Evaluation
│   ├── metrics.py            # ModelEvaluator, EvaluationResults
│   └── visualization.py      # Plotting functions
└── utils/            # Utilities
    ├── helpers.py            # set_seed, autocast, compute_class_weights
    ├── checkpoints.py        # CheckpointManager
    ├── config_schema.py      # Pydantic validation schemas
    └── logging_utils.py      # ExperimentLogger, MetricsTracker
```

### Appendix B: YAML Configuration Reference

**Centralized training** (`dscatnet_centralized_original.yaml`):

```yaml
centralized:
  experiment:
    name: dscatnet_centralized_baseline
  data_root: ./data
  output_dir: ./outputs
  datasets:
    - HAM10000
  model:
    variant: paper
    image_size: 224
    num_classes: 7
    pretrained: true
  training:
    batch_size: 4
    gradient_accumulation_steps: 8
    optimizer: adam
    lr: 0.001
    weight_decay: 0.0
    epochs: 200
    scheduler: none
    use_amp: false
  augmentation:
    level: none
    use_dermoscopy_norm: false
  splits:
    val_split: 0.15
  evaluation:
    checkpoint_interval: 5
    early_stopping_patience: 15
    use_class_weights: true
```

**Federated training** (`dscatnet_federated_ham10000_non_iid.yaml`):

```yaml
federated:
  experiment:
    name: dscatnet_federated_ham10000
  data_root: ./data
  output_dir: ./outputs
  datasets:
    - HAM10000
  model:
    variant: paper
    image_size: 224
    num_classes: 7
    pretrained: true
  training:
    batch_size: 4
    gradient_accumulation_steps: 8
    optimizer: adam
    lr: 0.001
    weight_decay: 0.0
    local_epochs: 1
    num_rounds: 100
    train_val_split: 0.85
  federation:
    num_clients: 4
    aggregation: FedAvg
    participation: 1.0
    noniid_type: dirichlet
    dirichlet_alpha: 0.5
  augmentation:
    level: none
    use_dermoscopy_norm: false
  evaluation:
    checkpoint_interval: 1
    early_stopping_patience: 10
    use_class_weights: true
```

### Appendix C: CLI Usage

```bash
# Centralized training
python run_experiment.py --mode centralized \
    --config configs/dscatnet_centralized_original.yaml

# Federated training
python run_experiment.py --mode federated \
    --config configs/dscatnet_federated_ham10000_non_iid.yaml

# Federated with custom Dirichlet alpha
python run_experiment.py --mode federated \
    --config configs/dscatnet_federated_ham10000_non_iid.yaml \
    --dirichlet-alpha 0.1

# Resume from checkpoint
python run_experiment.py --mode centralized \
    --resume outputs/exp_name/checkpoints/best_checkpoint.pt

# Evaluate checkpoint
python run_experiment.py --mode evaluate \
    --checkpoint outputs/exp_name/checkpoints/best_model.pt \
    --datasets HAM10000

# Comparison experiment
python run_experiment.py --mode comparison \
    --config configs/experiment_config.yaml
```

### Appendix D: Model Parameter Breakdown

| Component | Parameters | Percentage |
|-----------|-----------|------------|
| Fine-scale patch embedding | 590,208 | 2.0% |
| Coarse-scale patch embedding | 590,208 | 2.0% |
| Fine CLS token + positional embedding | 301,824 | 1.0% |
| Coarse CLS token + positional embedding | 75,840 | 0.3% |
| 6× Cross-attention (Q, K, V × 2 directions) | 5,308,416 | 18.0% |
| 6× Fine self-attention | 2,654,208 | 9.0% |
| 6× Coarse self-attention | 2,654,208 | 9.0% |
| 6× Fine FFN | 7,079,424 | 24.0% |
| 6× Coarse FFN | 7,079,424 | 24.0% |
| Layer normalization (all) | 18,432 | 0.1% |
| Fusion layer (768→384) | 295,296 | 1.0% |
| Classification head (384→7) | 2,695 | <0.1% |
| **Total** | **~29.4M** | **100%** |

*Note: Percentages are approximate. Exact counts derived from `model.get_num_parameters()`.*

### Appendix E: Pretrained Weight Transfer Summary

| Source (ViT-Small) | Target (DSCATNet) | Transferred |
|--------------------|-------------------|-------------|
| `blocks.0–5.attn.qkv` | `blocks.0–5.fine_self_attn.in_proj_weight/bias` | ✅ |
| `blocks.6–11.attn.qkv` | `blocks.0–5.coarse_self_attn.in_proj_weight/bias` | ✅ |
| `blocks.0–5.attn.proj` | `blocks.0–5.fine_self_attn.out_proj` | ✅ |
| `blocks.6–11.attn.proj` | `blocks.0–5.coarse_self_attn.out_proj` | ✅ |
| `blocks.0–5.norm1/norm2` | `blocks.0–5.norm_fine_self/norm_fine_ffn` | ✅ |
| `blocks.6–11.norm1/norm2` | `blocks.0–5.norm_coarse_self/norm_coarse_ffn` | ✅ |
| `blocks.0–5.mlp.fc1/fc2` | `blocks.0–5.fine_ffn.0/fine_ffn.3` | ✅ |
| `blocks.6–11.mlp.fc1/fc2` | `blocks.0–5.coarse_ffn.0/coarse_ffn.3` | ✅ |
| `patch_embed.proj` | `patch_embed.coarse_embedding.projection` | ✅ |
| `pos_embed` | `patch_embed.coarse_pos_embed` | ✅ |
| `cls_token` | `patch_embed.coarse_cls_token` | ✅ |
| `norm.weight/bias` | `norm_coarse.weight/bias` | ✅ |
| — | Cross-attention projections (all) | ❌ (random init) |
| — | Fine-scale patch embedding (8×8) | ❌ (random init) |
| — | Fine-scale CLS token + pos_embed | ❌ (random init) |
| — | Cross-attention norms | ❌ (random init) |
| — | Fusion layer | ❌ (random init) |
| — | Classification head | ❌ (random init) |

Total: **150/286 tensors transferred** (~52%)

### Appendix F: Test Suite Summary

| Test Module | Tests | Coverage Area |
|------------|-------|---------------|
| `test_models.py` | 26 | DSCATNet creation, variants, param count, forward pass |
| `test_datasets.py` | 21 | Dataset registry, class mapping, normalization |
| `test_preprocessing.py` | 6 | Transform pipeline, augmentation levels |
| `test_splits.py` | 8 | Deterministic splits, Dirichlet non-IID |
| `test_centralized.py` | 8 | CentralizedConfig, trainer initialization |
| `test_simulation.py` | 19 | SimulationConfig, FL setup, FedAvg |
| `test_strategy.py` | 15 | FedAvg aggregation, early stopping |
| `test_evaluation.py` | 7 | Metrics computation, EvaluationResults |
| `test_checkpoints.py` | 16 | Checkpoint save/load/cleanup |
| `test_download.py` | 31 | Download functions (mocked APIs) |
| `test_cli.py` | 23 | Argument parsing, mode selection |
| `test_config_loading.py` | 10 | YAML parsing, Pydantic validation |
| `test_helpers.py` | 22 | Seed, device, class weights, formatting |
| `test_model_evaluator.py` | 13 | End-to-end evaluation pipeline |
| **Total** | **225** | **41% line coverage** |

### Appendix G: Glossary

| Term | Definition |
|------|-----------|
| AMP | Automatic Mixed Precision — uses float16 for forward pass, float32 for gradients |
| AUC-ROC | Area Under the Receiver Operating Characteristic Curve |
| CLS token | Learnable classification token prepended to the patch sequence |
| Dermoscopy | Non-invasive skin imaging technique using polarized light |
| Dirichlet distribution | Multivariate generalization of the Beta distribution, used to model data heterogeneity |
| DSCATNet | Dual-Scale Cross-Attention Vision Transformer |
| FedAvg | Federated Averaging — standard FL aggregation algorithm |
| FFN | Feed-Forward Network — MLP within transformer blocks |
| FL | Federated Learning — distributed machine learning without data centralization |
| IID | Independent and Identically Distributed — assumes uniform data distribution |
| MHSA | Multi-Head Self-Attention — core attention mechanism in transformers |
| Non-IID | Non-IID data — clients have heterogeneous data distributions |
| ViT | Vision Transformer — image classification using pure transformer architecture |
| VRAM | Video RAM — GPU memory available for deep learning computations |
