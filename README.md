---
title: Grad-CAM vs HiResCAM Faithfulness Explorer
emoji: 🧠
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: 1.28.0
app_file: app.py
pinned: false
license: apache-2.0
---

# Grad-CAM vs HiResCAM Faithfulness Explorer

This interactive demo accompanies the research paper:

**"Faithfulness of Grad-CAM vs. HiResCAM on Sparse Data: A Comparative Study on the Quick Draw Dataset"**

## What This Demo Does

- Loads a pre-trained CNN model on 5 Quick Draw classes (airplane, bicycle, bus, car, train)
- Generates Grad-CAM and HiResCAM heatmaps for user-selected doodle images
- Compares faithfulness using deletion/insertion tests and AOPC metrics
- Visualizes results with interactive controls

## Research Findings

- **HiResCAM outperforms Grad-CAM** across all faithfulness metrics:
  - 52.5% better Deletion AUC
  - 36.4% better Insertion AUC
  - 56.2% better Deletion AOPC
  - 53.2% better Insertion AOPC

- **Methodological contributions:**
  - Percentile-based adaptation for sparse data
  - Replicate padding eliminates boundary artifacts
  - Controlled experimental design (3 conditions tested)

## How to Use

1. Select a sample image from the Quick Draw dataset
2. View Grad-CAM and HiResCAM heatmaps side-by-side
3. Compare faithfulness metrics and perturbation curves
4. Understand why HiResCAM provides more faithful explanations

## Try It Live

**👉 [Launch the Demo](https://huggingface.co/spaces/yourusername/gradcam-hirescam-faithfulness)**

## Code & Resources

- **GitHub Repository:** [Link to your repo]
- **Research Paper:** [Link to arXiv/preprint]
- **Dataset:** Quick Draw (Google Creative Lab)

## Citation

If you use this demo or findings in your research, please cite:

```bibtex
@article{yourname2026faithfulness,
  title={Faithfulness of Grad-CAM vs. HiResCAM on Sparse Data: A Comparative Study on the Quick Draw Dataset},
  author={[Your Name]},
  journal={[Journal/Conference Name]},
  year={2026}
}