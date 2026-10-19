# 🌿 Plant Disease Detection with Deep Learning

Leaf disease classifier built with **transfer learning** on **ResNet18**, trained on the **PlantVillage** dataset (54,305 images, 38 classes, 14 plant species).

**Live demo:** https://huggingface.co/spaces/fatybk/plant-disease-detection

## Results

| Metric | Test set (8,146 images) |
| --- | --- |
| Accuracy | 98.5% |
| Macro F1-score | 0.981 |
| Macro precision | 0.983 |
| Macro recall | 0.980 |

## Method

1. **Data**: PlantVillage (color version), stratified split 70% / 15% / 15%.
2. **Augmentation**: random resized crop, horizontal/vertical flips, rotation (±20°), color jitter.
3. **Model**: ResNet18 pre-trained on ImageNet, final layer replaced by a 38-class layer.
4. **Training in two phases** (Adam, CrossEntropyLoss, batch size 32, mixed precision, Tesla T4 GPU):
   - Phase 1: frozen backbone, final layer only, lr = 1e-3, 3 epochs.
   - Phase 2: full fine-tuning, lr = 1e-4, 3 epochs.
5. **Deployment**: Gradio interface (tested in Colab), then ONNX export and in-browser inference with ONNX Runtime Web on a static Hugging Face Space.

## Repository structure

| Folder | Content |
| --- | --- |
| `notebook/` | Full Colab notebook: EDA, training, evaluation, Gradio app |
| `app_gradio/` | Gradio application (`app.py`) and its requirements |
| `web/` | Deployed web app: `index.html`, ONNX model, labels |
| `models/` | Trained PyTorch weights (`.pth`) |

## Run locally

**Gradio app:**

    cd app_gradio
    pip install -r requirements.txt
    python app.py

The Gradio app loads the trained model from `models/meilleur_modele.pth`.

**Web app:** serve the `web/` folder with any static server, for example:

    cd web
    python -m http.server 8000

Then open http://localhost:8000.

## Limitations

- Trained on lab photos (single leaf, plain background): accuracy drops on field photos (domain shift).
- Closed-set classifier: any image gets one of the 38 labels, even unknown plants.

## Dataset

PlantVillage on Kaggle: https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset

## Tools

Python · PyTorch · torchvision · scikit-learn · Gradio · ONNX · Hugging Face Spaces · Google Colab
