# Predicting-Hypotension-in-ICU-Patients-Receiving-Vasopressor-Therapy-BA878-2025-team5

> **Course:** BA878: Machine Learning and Data Infrastructure in Healthcare (Boston University, Fall 2025)  
> **Team 5:** Aaryan Bammi, Pushpraj Singh, Yimeng Dong

## Project Overview
Hypotension (low blood pressure) in Intensive Care Unit (ICU) patients, particularly those on vasopressor therapy, is a life-threatening event. Traditional scoring systems often fail to capture the dynamic temporal patterns preceding these events.

This project leverages the **MIMIC-IV database** to build a Deep Learning pipeline. Our goal is to predict the onset of hypotension within the first **24 hours** of vasopressor initiation using **time-series clinical data**.

## Methodology
We developed a hybrid deep learning architecture that processes two types of data inputs:
1.  **Dynamic Temporal Data:** Time-series vitals (e.g., Heart Rate, Blood Pressure) processed via **GRU (Gated Recurrent Units)** and **Transformers**.
2.  **Static Clinical Data:** Patient demographics (e.g., Age, Gender) processed via dense layers.

**Key Techniques:**
* **Data Engineering:** Preprocessing raw MIMIC-IV data into time-series tensors using a 10-minute sliding window.
* **Imbalance Handling:** Utilized Class Weights to handle the skewed distribution of hypotensive events.
* **Architectures:** Comparison between GRU, Hybrid Transformer, and Ensemble models.
* **Fairness Audit:** Evaluated model performance across gender and racial subgroups.

## Repository Structure

- **GRU_Model.ipynb**: [Main Model] Implementation of the GRU architecture, evaluation, and fairness audit.
- **transformer.ipynb**: Implementation of the Hybrid Transformer model using MultiHeadAttention.
- **Tensor_V_5.ipynb**: Data Preprocessing: Converting raw MIMIC-IV CSVs into time-series tensors.
- **HC Final Report.pdf**: Detailed project report covering background, methods, and full results.
- **README.md**: Project documentation.

## Results
Our **GRU model** achieved the best performance, demonstrating robust predictive capability for clinical decision support.

| Model | AUC Score |
|-------|-----------|
| **GRU (Best)** | **0.9142** |
| Transformer | 0.8784 |

*Key finding: The GRU model effectively captures temporal dependencies in vital signs, significantly outperforming traditional baseline methods.*

## Requirements & Usage

### Prerequisites
* Python 3.8+
* TensorFlow / Keras
* Pandas, NumPy, Scikit-learn
* Access to MIMIC-IV Dataset (Credentialed access required via PhysioNet)

### How to Run
1.  **Data Preparation:** Run `Tensor_V_5.ipynb` to generate the preprocessed tensor files (`tensor_10min_24h.pkl`). *Note: You must have the raw MIMIC-IV files locally.*
2.  **Train Transformer:** Run `transformer.ipynb` to experiment with the attention-based architecture.
3.  **Train & Evaluate GRU:** Run `GRU_Model.ipynb` to train the final model and generate ROC curves and Fairness plots.

## Acknowledgments
* **MIMIC-IV Database:** Johnson, A. E. W., et al. (2020).
* **Boston University Questrom School of Business** for the course guidance.
