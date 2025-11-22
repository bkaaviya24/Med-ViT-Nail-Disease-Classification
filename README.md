Med-ViT: Vision Transformer for Nail Disease Classification and Severity Assessment

Official PyTorch implementation of "Vision Transformer-Based Classification of Nail Diseases and Severity Assessment from Clinical Images"

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Overview
Med-ViT is a Vision Transformer-based framework for automated nail disease classification and severity assessment. The model achieves 92.61% accuracy in disease classification and 96.88% accuracy in severity assessment while using only 0.0018% trainable parameters through strategic parameter freezing.

Key Features
- High Accuracy: 92.61% disease classification, 96.88% severity assessment
- Efficient Architecture: 99.998% parameter reduction (only 1,538-3,076 trainable parameters)
- Dual-Task Learning: Simultaneous disease classification and severity grading
- Resource-Efficient: 76.8% faster training, 74.2% lower memory usage
- Interpretable: Score-CAM visualizations for clinical validation

Dataset
The model is trained on 2,076 high-resolution nail images across 4 categories:
- Onychomycosis (Fungal Infection): 26.7%
- Psoriasis: 24.6%
- Nutritional Deficiency: 23.8%
- Healthy Nails: 22.4%

Dataset Access: [Kaggle Nail Disease Dataset](https://www.kaggle.com/datasets/nail-diseases)

Installation
```bash
# Clone repository
git clone https://github.com/bkaaviya24/Med-ViT-Nail-Disease-Classification.git
cd Med-ViT-Nail-Disease-Classification

# Install dependencies
pip install -r requirements.txt
```

Results
Disease Classification Performance
|        Class         | Precision (%) | Recall (%) | F1-Score (%) |
|----------------------|---------------|------------|--------------|
| Healthy              | 95.65         | 95.65      | 95.65        |
| Fungal               | 92.98         | 94.64      | 93.81        |
| Nutrition Deficiency | 91.84         | 90.00      | 90.91        |
| Psoriasis            | 94.12         | 89.47      | 91.74        |
| Overall              | 93.65         | 92.44      | 93.03        |

Comparison with State-of-the-Art
|      Model      |        Parameters       | Accuracy (%) | Training Time (min/epoch) |
|-----------------|-------------------------|--------------|---------------------------|
| VGG19           | 143.7M                  | 87.42        | 28.7                      |
| ResNet50        | 25.6M                   | 88.67        | 18.4                      |
| DenseNet121     | 7.98M                   | 91.23        | 13.8                      |
| EfficientNet-B3 | 12.2M                   | 90.45        | 14.2                      |
| Med-ViT (Ours)  | 85.8M (1,538 trainable) | 96.88        | 4.3                       |

Authors
- Kaaviya Balamurugan - bkaaviya24@gmail.com
- Supraja Venkatesan - Suprajavenky09@gmail.com
- Dr. Vanaja Selvaraj (Corresponding) - vanaja.bensingh@gmail.com
- Dr. Pathmanaban Pugazhendi - princepathu@gmail.com
- Justindhas Yesu Dhasan - justindhasy@gmail.com

Affiliation: Department of Computer Science and Engineering (AI & ML), Easwari Engineering College, Chennai, Tamil Nadu, India

Citation
```bibtex
@article{balamurugan2025medvit,
  title={Vision Transformer-Based Classification of Nail Diseases and Severity Assessment from Clinical Images},
  author={Balamurugan, Kaaviya and Venkatesan, Supraja and Selvaraj, Vanaja and Pugazhendi, Pathmanaban and Dhasan, Justindhas Yesu},
  journal={[Journal Name - Under Review]},
  year={2025}
}
```

License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Disclaimer
This model is intended for research purposes only and should not be used as a substitute for professional medical diagnosis.
