# Med-ViT: Vision Transformer for Nail Disease Classification and Severity Assessment

Official PyTorch implementation of **"Vision Transformer-Based Classification of Nail Diseases and Severity Assessment from Clinical Images"**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

##  Overview

Med-ViT is a Vision Transformer-based framework for automated nail disease classification and severity assessment. The model achieves **92.61% accuracy** in multi-class disease classification and **96.88% accuracy** in binary severity assessment while using only **0.0018% trainable parameters** through strategic parameter freezing.

##  Key Features

- **Dual-Task Learning**: Simultaneous disease classification (4 classes) and severity grading (2 classes)
- **High Accuracy**: 92.61% disease classification, 96.88% severity assessment
- **Efficient Architecture**: 99.998% parameter reduction (only 1,538-3,076 trainable parameters)
- **Resource-Efficient**: 76.8% faster training, 74.2% lower memory usage, 71.7% less energy consumption
- **Clinically Interpretable**: Score-CAM visualizations for transparent predictions

##  Dataset

The model is trained on **2,076 high-resolution nail images** across 4 diagnostic categories:

| Category | Distribution | Description |
|----------|--------------|-------------|
| **Onychomycosis (Fungal)** | 26.7% (554 images) | Fungal nail infections |
| **Psoriasis** | 24.6% (511 images) | Autoimmune nail condition |
| **Nutritional Deficiency** | 23.8% (493 images) | Vitamin/mineral deficiency |
| **Healthy Nails** | 22.4% (465 images) | Normal healthy nails |

**Dataset Split**: 80% training (1,618 images), 10% validation (202 images), 10% testing (203 images)

**Dataset Source**: [Kaggle Nail Disease Dataset](https://www.kaggle.com/datasets/nail-diseases)

##  Installation
```bash
# Clone repository
git clone https://github.com/bkaaviya24/Med-ViT-Nail-Disease-Classification.git
cd Med-ViT-Nail-Disease-Classification

# Install dependencies
pip install -r requirements.txt
```

### Requirements
- Python 3.8+
- PyTorch 2.0+
- CUDA-capable GPU (4GB+ VRAM recommended)
- See `requirements.txt` for full dependencies

##  Results

### Disease Classification Performance (4 Classes)

| Class | Precision (%) | Recall (%) | F1-Score (%) | Support |
|-------|--------------|------------|--------------|---------|
| Healthy | 95.65 | 95.65 | 95.65 | 46 |
| Fungal (Onychomycosis) | 92.98 | 94.64 | 93.81 | 56 |
| Nutrition Deficiency | 91.84 | 90.00 | 90.91 | 50 |
| Psoriasis | 94.12 | 89.47 | 91.74 | 51 |
| **Overall** | **93.65** | **92.44** | **93.03** | **203** |

**Overall Accuracy**: 92.61%

---

### Severity Assessment Performance (2 Classes)

| Class | Precision (%) | Recall (%) | F1-Score (%) | Support |
|-------|--------------|------------|--------------|---------|
| Mild | 96.36 | 98.15 | 97.25 | 108 |
| Severe | 97.56 | 95.24 | 96.39 | 95 |
| **Overall** | **96.96** | **96.70** | **96.82** | **203** |

**Overall Accuracy**: 96.88%

---

### Comparison with State-of-the-Art Models

| Model | Parameters | Trainable Params | Accuracy (%) | Training Time (min/epoch) |
|-------|------------|------------------|--------------|--------------------------|
| VGG19 | 143.7M | 143.7M | 87.42 | 28.7 |
| ResNet50 | 25.6M | 25.6M | 88.67 | 18.4 |
| DenseNet121 | 7.98M | 7.98M | 91.23 | 13.8 |
| EfficientNet-B3 | 12.2M | 12.2M | 90.45 | 14.2 |
| **Med-ViT (Ours)** | **85.8M** | **1,538-3,076** | **92.61 / 96.88** | **4.3** |

**Key Advantages**:
- **3.2-6.7× faster training** than competing models
- **74.2% lower GPU memory usage** (3.2GB vs 12.4GB)
- **71.7% less energy consumption** per epoch
- **99.998% parameter reduction** through transfer learning

### Model Architecture

Med-ViT uses Vision Transformer (ViT-B/16) with:

| Component | Specification |
|-----------|---------------|
| **Backbone** | ViT-B/16 (pretrained on ImageNet-21k) |
| **Total Parameters** | 85,803,270 |
| **Frozen Parameters** | 85,798,656 (99.995%) |
| **Trainable Parameters** | Disease: 3,076 / Severity: 1,538 |
| **Image Resolution** | 224×224 pixels |
| **Patch Size** | 16×16 pixels |
| **Transformer Blocks** | 12 layers |
| **Attention Heads** | 12 heads per block |
| **Hidden Dimension** | 768 |

**Key Innovation**: Strategic parameter freezing enables efficient fine-tuning while maintaining strong performance.

Usage

### Training Disease Classification Model
```bash
python Disease-Classification \
    --data_path ./data \
    --batch_size 32 \
    --epochs 30 \
    --lr 1e-3
```

### Training Severity Assessment Model
```bash
python Severity-Classification \
    --data_path ./data \
    --batch_size 32 \
    --epochs 30 \
    --lr 1e-3
```

### Repository Structure
```
Med-ViT-Nail-Disease-Classification/
├── Disease-Classification       # Multi-class disease classification (4 classes)
├── Severity-Classification      # Binary severity assessment (Mild/Severe)
├── README.md                    # Project documentation
├── requirements.txt             # Python dependencies
├── LICENSE                      # MIT License
└── .gitignore                  # Git ignore rules
```

## Authors

**Research Team**:
- **Kaaviya Balamurugan** - bkaaviya24@gmail.com
- **Supraja Venkatesan** - Suprajavenky09@gmail.com
- **Dr. Vanaja Selvaraj** (Corresponding Author) - vanaja.bensingh@gmail.com
- **Dr. Pathmanaban Pugazhendi** - princepathu@gmail.com
- **Justindhas Yesu Dhasan** - justindhasy@gmail.com

**Affiliation**: Department of Computer Science and Engineering (Artificial Intelligence and Machine Learning), Easwari Engineering College, Chennai, Tamil Nadu, India

## Citation

If you use this code in your research, please cite our paper:
```bibtex
@article{balamurugan2025medvit,
  title={Vision Transformer-Based Classification of Nail Diseases and Severity Assessment from Clinical Images},
  author={Balamurugan, Kaaviya and Venkatesan, Supraja and Selvaraj, Vanaja and Pugazhendi, Pathmanaban and Dhasan, Justindhas Yesu},
  journal={[Under Review]},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Vision Transformer implementation based on [timm](https://github.com/huggingface/pytorch-image-models)
- Dataset: [Kaggle Nail Disease Dataset](https://www.kaggle.com/datasets/nail-diseases)
- Pretrained weights: ImageNet-21k

## Contact

For questions, collaborations, or access to pre-trained weights:
- **Corresponding Author**: Dr. Vanaja Selvaraj (vanaja.bensingh@gmail.com)

## Disclaimer

This model is intended for **research purposes only** and should not be used as a substitute for professional medical diagnosis. Always consult qualified healthcare professionals for medical advice and treatment decisions.

---

**Keywords**: Deep Learning, Vision Transformer, Medical Image Analysis, Nail Disease Classification, Dermatology AI, Transfer Learning, Computer-Aided Diagnosis
```

---

## Summary:

**Extended Description to Use:**
```
Comprehensive documentation for Med-ViT project including overview, installation instructions, dataset information, performance metrics for both disease classification (92.61%) and severity assessment (96.88%), comparison with state-of-the-art models, and usage examples.
