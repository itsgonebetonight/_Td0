"""Predict script: given an image + metadata, return predicted probabilities and similarity match."""
import os
import argparse
import json
import numpy as np
import joblib

from .model import EmbeddingExtractor


def load_artifacts(artifacts_dir: str):
    clf = joblib.load(os.path.join(artifacts_dir, "classifier.joblib"))
    nn = joblib.load(os.path.join(artifacts_dir, "nn.joblib"))
    stock_enc = joblib.load(os.path.join(artifacts_dir, "stock_enc.joblib"))
    timeframe_enc = joblib.load(os.path.join(artifacts_dir, "timeframe_enc.joblib"))
    label_enc = joblib.load(os.path.join(artifacts_dir, "label_enc.joblib"))
    embeddings = np.load(os.path.join(artifacts_dir, "embeddings.npy"))
    # meta jsonl is optional
    meta_path = os.path.join(artifacts_dir, "meta.jsonl")
    meta = []
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            for line in f:
                meta.append(json.loads(line))
    return dict(clf=clf, nn=nn, stock_enc=stock_enc, timeframe_enc=timeframe_enc, label_enc=label_enc, embeddings=embeddings, meta=meta)


def predict(image_path: str, stock: str, timeframe: str, artifacts_dir: str, top_k: int = 5):
    art = load_artifacts(artifacts_dir)
    extractor = EmbeddingExtractor()
    emb = extractor.image_to_embedding(image_path)

    # prepare features as during training: concat embedding + encoded stock + timeframe
    try:
        stock_id = art["stock_enc"].transform([stock])[0]
    except Exception:
        # unseen stock -> map to -1
        stock_id = -1
    try:
        timeframe_id = art["timeframe_enc"].transform([timeframe])[0]
    except Exception:
        timeframe_id = -1

    Xq = np.concatenate([emb, np.array([float(stock_id), float(timeframe_id)])])
    probs = art["clf"].predict_proba(Xq.reshape(1, -1))[0]
    classes = art["label_enc"].inverse_transform(np.arange(len(probs))) if hasattr(art["label_enc"], "inverse_transform") else list(range(len(probs)))

    # similarity: use NN on raw embeddings
    dists, idxs = art["nn"].kneighbors(emb.reshape(1, -1), n_neighbors=top_k)
    dists = dists[0].tolist()
    idxs = idxs[0].tolist()
    # simple match score: inverse of mean distance
    mean_dist = float(np.mean(dists))
    match_score = 1.0 / (1.0 + mean_dist)

    neighbors = []
    for i, dist in zip(idxs, dists):
        item = {"index": int(i), "distance": float(dist)}
        if art["meta"]:
            item.update(art["meta"][i])
        neighbors.append(item)

    out = {"probs": {str(c): float(p) for c, p in zip(classes, probs)}, "match_score": match_score, "neighbors": neighbors}
    return out


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--stock", required=True)
    parser.add_argument("--timeframe", required=True)
    parser.add_argument("--artifacts", required=True, help="Path to artifacts directory (where train saved them)")
    args = parser.parse_args(argv)

    res = predict(args.image, args.stock, args.timeframe, args.artifacts)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
