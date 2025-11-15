Project: Chart-Image Outcome Predictor

This project provides a minimal pipeline to train a model that:
- extracts embeddings from trading chart images (using a pretrained ResNet)
- trains a classifier on those embeddings (plus simple metadata: stock, timeframe)
- builds a nearest-neighbor index for similarity / match scoring
- exposes a prediction script that returns class probabilities and a similarity score / closest examples

Assumptions & dataset format
- You have a CSV file with columns: image_path, stock, timeframe, label
  - image_path: path to the chart image file (PNG/JPG)
  - stock: stock symbol (string)
  - timeframe: timeframe identifier (e.g., "1m", "5m", "1h", "1d")
  - label: outcome class (e.g., "up", "down", "flat") or integer class

Example usage
1. Install dependencies (create a venv first):

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Train the model (example):

```powershell
python -m src.train --csv data/dataset.csv --outdir artifacts --epochs 1
```

3. Predict on a new chart image:

```powershell
python -m src.predict --image some_chart.png --stock AAPL --timeframe 1h --artifacts artifacts
```

Notes & next steps
- This scaffold uses PyTorch for embedding extraction and scikit-learn for the classifier/NN. If your dataset is large, switch the classifier to a scalable method or fine-tune the CNN.
- The code expects reasonable disk paths in the CSV. If your dataset structure differs, adapt `src/data.py`.

License: MIT
