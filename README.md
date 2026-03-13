# Gastric Sentinel 
**Advanced AI-driven diagnostic framework designed for early gastric cancer detection 
and lesion segmentation, leveraging a multi-modal fusion layer**.
To integrate diverse data sources like endoscopic images, CT/MRI scans, 
and clinical metadata.
<br>
# The multi-modal fusion layer 
Employs a **dual-branch architecture**_typically combining a CNN
based U-Net for local feature extraction from imaging modalities and a Transformer (e.g., Swin-Transformer) 
for capturing long-range dependencies—followed by a Feature Fusion (FF) module that uses linear Hadamard products
 and cross-modal attention to interactively merge features at multiple scales, enhancing lesion boundary precision.
<br>
<br>
<h1>Explainability Integration</h1>
<br>
<b>Grad-CAM</b> is incorporated post-fusion to generate class activation heatmaps, highlighting salient regions in input images
(e.g., tumor invasion depth in gastric walls) that drive predictions, thus providing visual interpretability for 
clinicians during T-staging or segmentation tasks.
<br>
<br>
<b>SHAP</b> Complements Grad-CAM by computing feature importance values across modalities, 
quantifying contributions from radiomic features(e.g., shape eccentricity, Haralick textures),
tabular data, and fused embeddings.
<br>
<br>
## Features
<br>
- ✅ Multi-modal: Endoscopy + CT/MRI + clinical data
<br>
- ✅ Precise lesion boundary detection
<br>
- ✅ Clinician-trusted explainability
<br>
- ✅ Production-ready architecture
