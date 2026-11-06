# Multi-Target UTS and EL Prediction

This project uses SPXY (Sample set Partitioning based on joint X-Y distances) to split the dataset and trains a fully connected neural network with a shared backbone and two output heads to predict the ultimate tensile strength (UTS) and elongation (EL) of alloys. It provides a complete workflow covering raw-data preparation, feature engineering, model training, held-out testing, inference on independent samples, and result visualization.

## Main Results

The dataset contains 248 samples. The fixed SPXY split produces 190 training samples, 21 validation samples, and 37 test samples, with approximately 15% of the data reserved for testing.

| Target | Test R² |    RMSE |     MAE |
| ------ | -------: | ------: | ------: |
| UTS    | 0.936489 | 76.4060 | 58.5950 |
| EL     | 0.867389 |  7.1357 |  4.0992 |

The UTS result exceeds the original R² acceptance threshold of 0.9137. The EL result is approximately 0.0127 below its original threshold of 0.8801; this version has been retained as an accepted result.

## Method Overview

1. Build 18 raw input features from 10 elemental compositions and 8 physicochemical descriptors.
2. Split the data into training, validation, and test sets using SPXY, which considers distances in both the input space X and output space Y.
3. Construct element-presence features, composition statistics, and second-order composition interactions, producing 86 candidate features.
4. Rank the second-order interaction terms by their training-set Pearson correlations and retain 55 final model inputs.
5. Standardize the inputs, UTS, and EL using training-set statistics, then select the model through validation-based early stopping across six fixed initializations.
6. Evaluate the selected model on the independent test set and apply the saved parameters to new samples.

See [WORKFLOW.md](WORKFLOW.md) for the complete method, training settings, and early-stopping details. See [DATA_FILES.md](DATA_FILES.md) for a full description of the project files.

## Project Structure

```text
ML/
├─ requirement.txt                   # Python dependencies
├─ README.MD                         # Project overview and usage
├─ WORKFLOW.md                       # Complete modeling workflow
├─ DATA_FILES.md                     # Directory and file descriptions
├─ 0_raw_data/                       # Raw data, descriptors, and final training data
├─ 1_SPXY/                           # SPXY split script and fixed indices
├─ 2_train/
│  ├─ 2_1_feature_engineering/       # Feature construction, selection, and correlation plot
│  └─ 2_2_model_training/            # Model, training scripts, parameters, and history
├─ 3_test/                           # Held-out test inference, metrics, and scatter plot
└─ 4_predict/                        # New-sample predictions and prediction-vs-experiment plot
```

## Requirements

- Python 3.10 (the current results were verified with Python 3.10)
- PyTorch 1.12 or later
- NumPy 1.x
- Pandas 1.5 or later
- Matplotlib 3.6 or later

Training and inference run on the CPU by default; CUDA is not required.

## Installation

Run the following commands from the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirement.txt
```

On Linux or macOS, activate the virtual environment with:

```bash
source .venv/bin/activate
```

## Reproducing the Complete Workflow

Run all commands below from the `ML` root directory.

### 1. Generate the Fixed SPXY Split

```powershell
python .\1_SPXY\spxy_split.py
```

By default, this command reads `0_raw_data/parameter.csv` and writes the training, validation, and test indices to `1_SPXY/split_indices.json`.

### 2. Train the Model

```powershell
python .\2_train\2_2_model_training\train_fcnn.py
```

The training script writes its outputs to `2_train/2_2_model_training/` and directly generates:

- `fcnn_model.pt`: model weights, selected feature indices, and standardization parameters;
- `metrics.json`: data-split information, candidate-model results, and final metrics.

To reproduce `training_history.csv` and `training_loss_curve.png` for the selected model, run:

```powershell
python .\2_train\2_2_model_training\export_training_history.py
```

### 3. Export Feature Results and the Correlation Plot

```powershell
python .\2_train\2_1_feature_engineering\make_feature_heatmaps.py
```

This command exports the 86 candidate features, the 55 selected features, their Pearson correlation coefficients, and a circular correlation plot containing the 18 raw features.

### 4. Run Test-Set Inference and Plot the Results

```powershell
python .\3_test\test_fcnn.py
python .\3_test\plot_test_predictions.py
```

The test script only reads the fixed test indices from `1_SPXY/split_indices.json`; it does not retrain the model or repartition the data. Row-level predictions and the scatter plot are saved in `3_test/`.

### 5. Predict New Data

```powershell
python .\4_predict\predict_fcnn.py .\path\to\pre.csv --output .\4_predict\pre_with_predictions.csv
```

The first 18 columns and their headers in the input CSV must match columns A:R in `0_raw_data/parameter.csv`. The output retains the original input columns and appends:

- `pre_UTS`
- `pre_EL`

If `4_predict/pre_vs_exp.csv` contains both predicted and experimental values, generate the comparison scatter plot with:

```powershell
python .\4_predict\plot_pre_vs_exp.py
```
