# Federated CNN for Plant Disease Classification (ResNet-18 + FedAvg)

**Academic Project — National Institute of Technology, Puducherry**
Dept. of CSE | Supervisor: Dr. M. Venkatesan, Associate Professor
Contributors: K S G S Lokesh (EE23B1027)

A privacy-preserving plant disease classifier that combines **Federated Learning (FedAvg)** with **transfer learning on ResNet-18**, trained across simulated distributed clients without ever centralizing raw image data — reaching **97% test accuracy** on the PlantVillage dataset.

---

## Problem & Motivation

Plant disease detection is central to precision agriculture — catching disease early cuts crop losses and reduces unnecessary pesticide use. CNNs are very good at this from leaf images, but the standard approach requires pooling every farm's images onto one server, which raises **privacy** and **bandwidth** concerns for real-world agricultural deployments.

This project asks: *can we train an accurate plant-disease classifier without collecting anyone's raw images?* The answer is Federated Learning — clients train locally and only share model weight updates, which a central server aggregates into a global model.

---

## Results

| Metric | Value |
|---|---|
| **Final global model accuracy** | **97%** on held-out PlantVillage test set |
| Top-5 accuracy | Notably higher than top-1 — correct class consistently in the model's top predictions |
| Per-class accuracy | Strong across most of the 38 categories; classes with more samples perform better |
| Confusion matrix | Misclassifications concentrated among visually similar disease patterns |

Evaluated after **5 FedAvg communication rounds** across **3 simulated clients**.

---

## Repository Structure

| File | Description |
|---|---|
| `Federated_CNN_PlantDisease.ipynb` | Main notebook — end-to-end pipeline: data loading, federated partitioning, local training, FedAvg aggregation, evaluation, and visualizations. |
| `Federated_CNN_PlantDisease.py` | Script version of the same pipeline. |
| `Federated_CNN_PlantDisease_PPT.pptx` | Project presentation covering motivation, literature survey, architecture, results, and future scope. |

---

## Method

### 1. Dataset
- **PlantVillage** (RGB version) — ~54,000 leaf images across **38 plant disease / healthy classes**, downloaded via KaggleHub
- Images resized to **224×224**, normalized with ImageNet statistics (mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`)
- Split **80/20** into train/test via `torch.utils.data.random_split`

### 2. Federated Partitioning
- Training set divided equally into **3 disjoint, non-overlapping subsets** — one per simulated client — using non-overlapping index ranges
- This models **horizontal federated learning**: each client holds different examples of the same feature/label space

### 3. Model
- **ResNet-18**, pre-trained on ImageNet, convolutional backbone **frozen**
- Final fully-connected layer replaced with a new linear layer for **38 output classes**
- Only this final layer is updated during local training — a lightweight fine-tuning approach well-suited to federated settings with limited local compute/epochs

### 4. Local Training
- Each client trains independently for **2 epochs**
- Optimizer: **Adam**, `lr = 0.001`
- Loss: **Cross-Entropy**

### 5. FedAvg Aggregation
- Server collects the locally trained weights from all 3 clients
- Computes the **arithmetic mean** of the weights to form the new global model
- Redistributes the updated global model to all clients
- Repeated for **5 communication rounds**

### 6. Evaluation
- Overall accuracy, top-5 accuracy, per-class accuracy
- Classification report and confusion matrix
- Training loss curves and sample prediction visualizations

---

## Tech Stack

- **Language:** Python 3.8+
- **Deep Learning:** PyTorch 1.12+, torchvision
- **Model:** ResNet-18 (`torchvision.models`, ImageNet pre-trained)
- **Libraries:** scikit-learn, KaggleHub, seaborn, matplotlib, numpy
- **Environment:** Google Colab / Jupyter Notebook

---

## Why This Approach

- **Privacy-preserving** — raw training data never leaves a client; only model weight updates are shared
- **Bandwidth-efficient** — transmitting model parameters is far cheaper than transmitting full image datasets
- **Scalable** — the FedAvg pipeline extends to more clients without redesign
- **Strong initialization** — ImageNet transfer learning compensates for the limited local epochs each client can afford
- **Thoroughly evaluated** — accuracy, top-5, per-class breakdown, and confusion matrix rather than a single headline number

---

## Related Work

This project builds on:
- **McMahan et al. (2017)** — introduced FedAvg, the communication-efficient federated aggregation algorithm used here
- **Mohanty et al. (2016)** — established PlantVillage as a benchmark, showing centralized CNNs exceed 99% accuracy under controlled conditions
- **He et al. (2016)** — ResNet, whose residual connections make deep transfer-learning backbones like ResNet-18 practical
- **Rieke et al. (2020)** — federated learning for privacy-preserving collaborative learning (originally healthcare, principles applied here to agriculture)
- **Liu et al. (2020)** — showed FL + transfer learning improves convergence, especially under non-IID client data

---

## Limitations & Future Scope

- **IID assumption:** clients currently hold random, identically-distributed slices of the data — real farms would see **non-IID** data (e.g., only 1–2 crop types per client). A natural next step is simulating that harder, more realistic setting.
- **Client count:** only 3 simulated clients; scaling to more clients would stress-test FedAvg's robustness.
- **Aggregation strategy:** exploring **FedProx**, **SCAFFOLD**, or **FedNova** for better convergence under heterogeneous data.
- **Deeper fine-tuning:** progressively unfreezing deeper ResNet-18 layers across rounds to push accuracy further.
- **Edge deployment:** compressing the global model for on-device inference (Raspberry Pi, mobile) via TensorFlow Lite or ONNX, enabling real-time field diagnosis.
- **Formal privacy guarantees:** integrating differential privacy and secure aggregation on top of basic data locality.

---

## Conclusion

This project shows federated learning is a practical, privacy-preserving path to accurate plant disease classification: aggregating locally-trained ResNet-18 weights from 3 clients over 5 FedAvg rounds — with no client ever sharing raw images — reaches 97% test accuracy, demonstrating that agricultural image classification at scale doesn't require centralizing sensitive farm data.
