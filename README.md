# 🔬 GastricSentinel

> **Multimodal AI for gastric cancer classification — fusing histopathology, genomics, and clinical data into a single explainable prediction.**

---

## 🧬 What is GastricSentinel?

**GastricSentinel** is a full-stack multimodal AI web application for gastric cancer classification. It fuses three distinct data modalities — histopathological images, genomic profiles, and clinical records — through a dedicated **fusion layer** to produce **8-class probability predictions**. The system also generates explainability outputs via **Grad-CAM** (visual heatmaps over tissue images) and **SHAP values** (feature importance across genomic and clinical inputs), making every prediction interpretable.

Gastric cancer is one of the leading causes of cancer-related deaths worldwide. Early and accurate classification is critical, yet manual histopathological screening is slow, labor-intensive, and prone to human error. GastricSentinel aims to assist pathologists and researchers with a fast, multimodal, and explainable second opinion.

---

## ✨ Features

- 🖼️ **Histopathological Image Analysis** — deep learning over gastric tissue images via GasHisSDB-trained CNN
- 📄 **Downloadable PDF Report** — scan a tissue image and download a full report including predictions, risk score, and Grad-CAM visualizations
- 🧬 **Genomic & Clinical Data Integration** — incorporates TCGA-STAD genomic profiles and clinical records
- 🔀 **Fusion Layer** — dedicated multimodal fusion layer combines image, genomic, and clinical embeddings before classification
- 🎯 **8-Class Probability Output** — predicts across 8 distinct gastric cancer classes with per-class confidence scores
- ⚠️ **Risk Score** — synthesizes model outputs into an overall patient risk score for quick clinical triage
- 🔥 **Grad-CAM Visualizations** — gradient-weighted heatmaps highlight the tissue regions driving each prediction
- 📊 **SHAP Explainability** — SHAP values reveal which genomic and clinical features most influence the model's decision
- 🤖 **Integrated Chatbot** — conversational assistant to guide users through the platform and help interpret results
- 🗄️ **MongoDB Atlas** — cloud-hosted NoSQL database for storing user data, predictions, and session records
- 🌐 **Full-Stack Web App** — clean, responsive frontend with a Python-powered REST API backend
- 📁 **Modular Architecture** — clearly separated `frontend`, `backend`, and `training` modules

---

## 🗂️ Project Structure

```
GastricSentinel/
├── frontend/          # Web UI — HTML, CSS, JavaScript
│   └── ...
├── backend/           # REST API server — Python
│   └── ...
├── training/          # Model training scripts & notebooks
│   └── ...
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- Node.js (if using a bundler for frontend)
- pip

### 1. Clone the Repository

```bash
git clone https://github.com/adyashaexe/GastricSentinel.git
cd GastricSentinel
```

### 2. Set Up the Backend

```bash
cd backend
pip install -r requirements.txt
python app.py
```

The backend API will start at `http://localhost:5000` (or as configured).

### 3. Launch the Frontend

Open `frontend/index.html` directly in your browser, or serve it with any static file server:

```bash
# Using Python's built-in server
cd frontend
python -m http.server 3000
```

Then visit `http://localhost:3000`.

---

## 🤖 Model Architecture

GastricSentinel uses a **multimodal deep learning architecture** that processes three input streams in parallel before combining them through a fusion layer for final classification.

```
 Histopathology Images  ──►  CNN Encoder  ──────────────────┐
                                                             │
 Genomic Profiles       ──►  Genomic Encoder  ──────────►  Fusion Layer  ──►  8-Class Output
                                                             │
 Clinical Records       ──►  Clinical Encoder  ────────────┘
```

### Output Classes

The model predicts probabilities across **8 gastric cancer classes**, providing fine-grained classification beyond a simple normal/abnormal split.

### Explainability

| Method | Modality | What it shows |
|--------|----------|---------------|
| **Grad-CAM** | Image | Heatmap overlay highlighting the tissue regions most influential to the prediction |
| **SHAP** | Genomic & Clinical | Feature importance scores showing which genes and clinical variables drive the model's decision |

Training scripts and notebooks are available in the `/training` directory.

---

## 📦 Datasets

GastricSentinel draws on three categories of data: histopathological images, genomic profiles, and structured clinical records. All datasets listed below are publicly available for research use.

---

### 🔬 Histopathological Image Data

#### GasHisSDB — Gastric Histopathology Sub-size Image Database
> The primary dataset used for image classification training.

- **Size:** 245,196 labeled sub-size tissue images (Normal / Abnormal)
- **Sub-databases:** 80×80 px, 120×120 px, 160×160 px patch sizes
- **Source:** Northeastern University, China Medical University
- **Paper:** [arXiv:2106.02473](https://arxiv.org/abs/2106.02473) · [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0010482521010015)
- **Download:** [Gitee Repository (NEUhwm/MIaMIA-Open-Data-Gastric-Histopathology-Image)](https://gitee.com/neuhwm/GasHisSDB) *(contact authors for full dataset)*

---

### 🧬 Genomic Data

#### TCGA-STAD — Genomic Profiles (GDC Data Portal)
> Multi-dimensional genomic characterization of ~450 gastric adenocarcinoma cases, including RNA-seq gene expression, somatic mutations, copy number variation, miRNA, and DNA methylation.

- **Samples:** ~450 stomach adenocarcinoma cases
- **Data types:** RNA-Seq, WXS somatic mutations, miRNA-Seq, CNV, DNA methylation
- **Access:** Open Access (no registration required) + Controlled Access (dbGaP for raw sequence files)
- **Download:** [portal.gdc.cancer.gov/projects/TCGA-STAD](https://portal.gdc.cancer.gov/projects/TCGA-STAD)
- **Publication:** [NCI GDC STAD 2014](https://gdc.cancer.gov/about-data/publications/stad_2014)
- **Interactive browser:** [cBioPortal](https://www.cbioportal.org/) → search "TCGA Stomach"

---

### 🏥 Clinical Data

#### TCGA-STAD — Clinical Supplement (GDC Data Portal)
> Patient-level clinical records matched to the TCGA-STAD cohort, including staging, survival outcomes, treatment history, and biospecimen metadata.

- **Samples:** ~450 stomach adenocarcinoma cases
- **Data types:** Clinical XML, survival data, demographic records, biospecimen annotations
- **Access:** Free, no registration required
- **Download:** [portal.gdc.cancer.gov/projects/TCGA-STAD](https://portal.gdc.cancer.gov/projects/TCGA-STAD) → *Clinical* tab
- **Interactive browser:** [cBioPortal](https://www.cbioportal.org/) → search "TCGA Stomach"

---

> **Note:** When using TCGA data, please cite the TCGA Research Network: *"The results shown here are in whole or in part based upon data generated by the TCGA Research Network: https://www.cancer.gov/tcga."*

---

## 🖥️ Tech Stack

| Layer           | Technology                            |
|-----------------|---------------------------------------|
| Frontend        | HTML · CSS · JavaScript               |
| Backend         | Python · Flask (or FastAPI)           |
| Database        | Supabase(PostgresSQL)                 |
| ML/AI           | TensorFlow / PyTorch · Fusion Layer   |
| Explainability  | Grad-CAM · SHAP                       |
| Chatbot         | Conversational AI Assistant           |
| Datasets        | GasHisSDB · TCGA-STAD                 |

---

## 📸 Usage

1. Open the app in your browser
2. Upload a histopathological tissue image (`.jpg`, `.png`)
3. Provide genomic and clinical data inputs
4. Click **Analyze**
5. Receive:
   - **8-class probability scores** for gastric cancer classification
   - **Overall risk score** synthesized from all modalities
   - **Grad-CAM heatmap** overlaid on the tissue image
   - **SHAP values** showing feature importance across genomic and clinical inputs
6. Click **Download Report** to export a full PDF summary of the analysis

---

## ⚠️ Disclaimer

GastricSentinel is a **research and educational tool**. It is **not** intended for clinical diagnosis or medical decision-making. Always consult a qualified medical professional for any health-related concerns.

---

## 🤝 Contributing

Contributions are welcome! To get started:

1. Fork the repo
2. Create a new branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m 'Add your feature'`
4. Push to the branch: `git push origin feature/your-feature`
5. Open a Pull Request

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

<p align="center">
  Made with ❤️ to help in the fight against gastric cancer.
</p>
