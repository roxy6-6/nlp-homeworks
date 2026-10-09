from pathlib import Path
import json
import random

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split


SEED = 42
ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "nyt.csv"
OUT_DIR = ROOT / "task1_outputs"


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)


def main() -> None:
    set_seed()
    OUT_DIR.mkdir(exist_ok=True)

    data = pd.read_csv(DATA_PATH, usecols=["text", "label"])
    data = data.dropna(subset=["text", "label"]).reset_index(drop=True)
    data["text"] = data["text"].astype(str)
    data["label"] = data["label"].astype(str)

    # 为两种文本表示方法生成并统一使用同一个固定的分层数据划分。
    all_idx = np.arange(len(data))
    train_idx, temp_idx = train_test_split(
        all_idx, test_size=0.20, random_state=SEED,
        stratify=data["label"].to_numpy()
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.50, random_state=SEED,
        stratify=data.loc[temp_idx, "label"].to_numpy()
    )
    split = {"train": train_idx.tolist(), "validation": val_idx.tolist(), "test": test_idx.tolist()}
    with (OUT_DIR / "split_indices.json").open("w", encoding="utf-8") as f:
        json.dump(split, f, ensure_ascii=False, indent=2)

    texts = data["text"].to_numpy()
    labels = data["label"].to_numpy()
    results = {}

    for name, binary in (("binary_bow", True), ("word_frequency", False)):
        # 对所有数据子集使用一致的 CountVectorizer 分词方式，且只使用训练文档拟合词表。
        vectorizer = CountVectorizer(binary=binary, lowercase=True)
        x_train = vectorizer.fit_transform(texts[train_idx])
        x_val = vectorizer.transform(texts[val_idx])
        x_test = vectorizer.transform(texts[test_idx])

        # lbfgs 原生支持 NYT 数据集的三个类别（多分类设置）。
        clf = LogisticRegression(max_iter=1000, random_state=SEED, solver="lbfgs") # 创建逻辑回归分类器
        clf.fit(x_train, labels[train_idx])
        pred = clf.predict(x_test)

        result = {
            "representation": name,
            "vocabulary_size": int(len(vectorizer.vocabulary_)),
            "train_documents": int(len(train_idx)),
            "validation_documents": int(len(val_idx)),
            "test_documents": int(len(test_idx)),
            "accuracy": float(accuracy_score(labels[test_idx], pred)),
            "macro_f1": float(f1_score(labels[test_idx], pred, average="macro")),
            "classification_report": classification_report(
                labels[test_idx], pred, output_dict=True, zero_division=0
            ),
        }
        results[name] = result
        print(f"{name}: Accuracy={result['accuracy']:.6f}, Macro-F1={result['macro_f1']:.6f}")
        print(classification_report(labels[test_idx], pred, digits=4, zero_division=0))

    with (OUT_DIR / "results.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nSaved results to {OUT_DIR / 'results.json'}")


if __name__ == "__main__":
    main()
