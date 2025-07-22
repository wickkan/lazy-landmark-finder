# Lazy Landmark Finder

Train a model to recognize famous landmarks from the laziest angles (low effort, blurry shots, bad lighting) using transfer learning on a pre-trained CNN (e.g., ResNet or MobileNet). Fine-tune with the Google Landmarks Dataset v2 and deploy with OpenCV.

---

## Project Structure

```
lazy-landmark-finder/
  data/           # Scripts and data download helpers
  notebooks/      # Jupyter notebooks for exploration and prototyping
  scripts/        # Utility scripts (preprocessing, training, etc.)
  src/            # Source code (models, dataloaders, training, etc.)
  models/         # Trained model weights/checkpoints
  outputs/        # Predictions, logs, and results
  configs/        # Experiment/config files
  tests/          # Unit/integration tests
  docs/           # Additional documentation
  README.md
  requirements.txt
  .gitignore
```

---

## Dataset: Google Landmarks Dataset v2

This project uses the [Google Landmarks Dataset v2 (GLDv2)](https://github.com/cvdfoundation/google-landmark), a large-scale benchmark for instance-level recognition and retrieval.

- **Dataset webpage:** [Explore visually](https://storage.googleapis.com/gld-v2/web/index.html)
- **Download instructions:**  
  To download the dataset, use the provided script:

  ```bash
  mkdir train && cd train
  bash ../data/download-dataset.sh train 499
  ```

  Replace `train` and `499` with `test 19` or `index 99` for other splits.

- **License:**  
  The annotations are licensed by Google under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).  
  The images are publicly available on the web and may have different licenses. Please verify the license for each image yourself.

- **Citation:**  
  If you use this dataset, please cite:

  ```
  @inproceedings{weyand2020GLDv2,
    author = {Weyand, T. and Araujo, A. and Cao, B. and Sim, J.},
    title = {{Google Landmarks Dataset v2 - A Large-Scale Benchmark for Instance-Level Recognition and Retrieval}},
    year = {2020},
    booktitle = {Proc. CVPR},
  }
  ```

- **More info:**  
  See the [official dataset repo](https://github.com/cvdfoundation/google-landmark) for details, papers, and license information.

---

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Download the dataset:**
   See the instructions above or in `data/download-dataset.sh`.

---

## Usage

- Use the scripts in `src/` and `notebooks/` to train and evaluate models.
- Store trained models in `models/` and results in `outputs/`.
- Add new configs to `configs/` for experiment management.

---

## License

This project is for research and educational purposes. See dataset license above for data usage.

## Example Results

The current prototype was trained on a small subset of the Google Landmarks Dataset v2 (only `images_000.tar`, ~259 images/classes) for 5 epochs using a pre-trained ResNet18.

**Evaluation on the training subset:**

- Top-1 Accuracy: 0.35%
- Top-5 Accuracy: 0.82%

> **Note:**  
> These results are expectedly low due to the extremely limited data and lack of a validation split. For meaningful results, train on a larger portion of the dataset and consider adding data augmentations and a validation set.

---

## Next Steps

- Increase the dataset size by extracting more tar files.
- Add data augmentations to improve generalization.
- Implement a validation split for more realistic evaluation.
- Experiment with different architectures and hyperparameters.
