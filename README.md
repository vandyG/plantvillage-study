# Plant Disease Detection Study

Transfer learning study that compares twelve pretrained CNN and modern architectures for binary plant disease detection (healthy vs diseased). Models are trained on PlantVillage with augmentation for balance and tested for cross-dataset generalization on PlantDoc.

## What is in this repo
- Paper sources in `latex/` summarizing the research question, methods, and results.
- Training pipelines:
	- `notebooks/benchmark.py`: streaming TensorFlow data into PyTorch for memory-efficient benchmarking of multiple models.
	- `notebooks/pytorch_model.ipynb`: frozen-backbone PyTorch training on preprocessed NumPy arrays.
- Cross-dataset testing: `notebooks/test_plant_doc.py` evaluates saved checkpoints on PlantDoc and reports metrics and plots.
- Outputs and checkpoints: `models/` for weights, `output/` and `reports/` for evaluation JSON and charts.

## Datasets
- **PlantVillage**: 54,303 leaf images across 14 crops with multiple disease classes. Images are captured in controlled settings, so augmentation is used to improve diversity and address class imbalance.
- **PlantDoc**: 2,598 real-world images across 27 folders. For this study folders with only the crop name plus "leaf" are mapped to healthy (0); folders containing disease keywords (rust, scab, spot, blight, mildew, bacterial, virus, mosaic, rot) are mapped to diseased (1). Test split used in scripts contains 236 images (90 healthy, 146 diseased).

## Models compared
Classical: AlexNet, VGG16.
Lightweight: MobileNetV2, ShuffleNetV2, SqueezeNet1_0, MnasNet1_0, EfficientNet-Lite4.
Modern: ResNet50, DenseNet121, Xception, InceptionV4, ConvNeXt Base, plus any timm model supported by the registry in `benchmark.py`.

## Training pipelines
- **Streaming benchmark (`benchmark.py`)**
	- Loads PlantVillage via TensorFlow Datasets, applies class-specific augmentation, maps multi-class labels to binary, and streams batches to PyTorch through an `IterableDataset` wrapper.
	- Transfer learning with ImageNet weights, frozen backbones, BCEWithLogits loss, Adam (lr=1e-4), ReduceLROnPlateau scheduler, mixed precision, and optional gradient accumulation. Early stopping patience 2. Default batch sizes are architecture dependent.
	- Saves per-model metrics, ROC data, and checkpoints to `data/output/` and `models/`.
- **Frozen-backbone NumPy training (`pytorch_model.ipynb`)**
	- Uses preprocessed arrays in `notebooks/preprocessed_numpy/` with `DataLoader` batching (batch size 32, epochs 5, patience 2).
	- Supports torchvision backbones and timm models with optional backbone freezing. Tracks loss and accuracy each epoch, saves best weights to `models/` and a consolidated history JSON to `notebooks/results/all_models_training_history.json`.

## Cross-dataset testing (`test_plant_doc.py`)
- Loads all checkpoints in `models/`, rebuilds the matching architecture, and evaluates on PlantDoc with a consistent transform (resize to 224, ImageNet mean/std).
- Computes accuracy, precision, recall, F1, AUC, ROC curves, and confusion matrices; measures inference speed; exports combined JSON plus comparison plots to `data/output/`.

## Key findings (from paper)
- VGG16 reached the highest PlantVillage AUC-ROC (0.9990) and 97.05 percent accuracy; EfficientNet-Lite4 achieved the top PlantVillage accuracy (98.20 percent).
- ConvNeXt Base generalized best to PlantDoc (83.90 percent) despite ranking lower on PlantVillage.
- AlexNet offers the best speed to accuracy tradeoff: 96.53 percent on PlantVillage, 81.36 percent on PlantDoc, and fastest inference among tested models.
- Lightweight models overfit to PlantVillage backgrounds; MobileNetV2 and ShuffleNetV2 fell to 44.49 percent and 40.25 percent accuracy on PlantDoc.
- ResNet50 collapsed on PlantDoc (32.63 percent, AUC below 0.5), highlighting the need for cross-dataset checks.

## How to reproduce
1) Create a Python environment (see `pyproject.toml` for dependencies) and install PyTorch, torchvision, timm, TensorFlow, scikit-learn, matplotlib, seaborn, and tqdm.
2) Download PlantVillage (via `tfds`) and PlantDoc (place under `data/plant_doc/` in the same folder structure as the dataset archive). Ensure PlantDoc test split folders remain intact for keyword-based binary labeling.
3) Run benchmarking: `python notebooks/benchmark.py` to train and compare registered models with streaming augmentation.
4) Run the frozen-backbone notebook: open `notebooks/pytorch_model.ipynb` and execute to train selected torchvision/timm models on NumPy tensors.
5) Evaluate on PlantDoc: `python notebooks/test_plant_doc.py` to generate metrics and plots for all checkpoints in `models/`.

## Repository layout (selected)
- `notebooks/` scripts and exploratory notebooks
- `models/` trained weights
- `data/output/` evaluation JSON and charts
- `latex/` paper source
- `build.py`, `demo.sh`, `pyproject.toml`, `shell.nix` helper and environment files
