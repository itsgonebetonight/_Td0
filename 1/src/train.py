"""Train script: compute embeddings, train classifier and nearest-neighbor index."""
import os
import argparse
import json

import numpy as np
import joblib
from tqdm import tqdm

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder

from .data import load_dataset, validate_images_exist
from .model import EmbeddingExtractor


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="CSV dataset path with image_path,stock,timeframe,label")
    parser.add_argument("--outdir", required=True, help="Directory to save artifacts")
    parser.add_argument("--n_neighbors", type=int, default=5)
    args = parser.parse_args(argv)

    os.makedirs(args.outdir, exist_ok=True)

    df = load_dataset(args.csv)
    missing = validate_images_exist(df)
    if missing:
        print(f"Warning: {len(missing)} images missing. First 5: {missing[:5]}")

    extractor = EmbeddingExtractor()

    embeddings = []
    rows = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="embeddings"):
        p = row["image_path"]
        try:
            emb = extractor.image_to_embedding(p)
        except Exception as e:
            print(f"Skipping {p}: {e}")
            continue
        embeddings.append(emb)
        rows.append(row)

    if len(embeddings) == 0:
        raise RuntimeError("No embeddings were computed. Check dataset and image paths.")

    X_emb = np.stack(embeddings)
    meta = {"stock": [r["stock"] for r in rows], "timeframe": [r["timeframe"] for r in rows]}
    y_raw = [r["label"] for r in rows]

    # Encode metadata and labels
    stock_enc = LabelEncoder().fit(meta["stock"])
    timeframe_enc = LabelEncoder().fit(meta["timeframe"])
    label_enc = LabelEncoder().fit(y_raw)

    stock_feats = stock_enc.transform(meta["stock"]).reshape(-1, 1)
    timeframe_feats = timeframe_enc.transform(meta["timeframe"]).reshape(-1, 1)

    # Concatenate embedding with simple integer-encoded meta features
    X = np.concatenate([X_emb, stock_feats.astype(float), timeframe_feats.astype(float)], axis=1)
    y = label_enc.transform(y_raw)

    clf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
    print("Training classifier...")
    clf.fit(X, y)

    print("Fitting nearest-neighbors on raw embeddings...")
    nn = NearestNeighbors(n_neighbors=args.n_neighbors, metric="euclidean").fit(X_emb)

    # Save artifacts
    joblib.dump(clf, os.path.join(args.outdir, "classifier.joblib"))
    joblib.dump(nn, os.path.join(args.outdir, "nn.joblib"))
    joblib.dump(stock_enc, os.path.join(args.outdir, "stock_enc.joblib"))
    joblib.dump(timeframe_enc, os.path.join(args.outdir, "timeframe_enc.joblib"))
    joblib.dump(label_enc, os.path.join(args.outdir, "label_enc.joblib"))
    np.save(os.path.join(args.outdir, "embeddings.npy"), X_emb)
    # save meta rows as json-lines
    meta_rows = [{"image_path": r["image_path"], "stock": r["stock"], "timeframe": r["timeframe"], "label": r["label"]} for r in rows]
    with open(os.path.join(args.outdir, "meta.jsonl"), "w", encoding="utf-8") as f:
        for m in meta_rows:
            f.write(json.dumps(m) + "\n")

    print(f"Artifacts saved to {args.outdir}")


if __name__ == "__main__":
    main()
