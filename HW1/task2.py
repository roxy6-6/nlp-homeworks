from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import gensim.downloader as api
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score


SEED = 42
VECTOR_SIZE = 100
ROOT = Path(__file__).resolve().parent
NYT_PATH = ROOT / "nyt.csv"
AG_PATH = ROOT / "ag.csv"
SPLIT_PATH = ROOT / "task1_outputs" / "split_indices.json"
OUT_DIR = ROOT / "task2_outputs"
TOKEN_PATTERN = re.compile(r"[a-z]+(?:'[a-z]+)?")


def set_seed(seed: int = SEED) -> None:
    """固定 Python 和 NumPy 的随机种子。"""
    random.seed(seed)
    np.random.seed(seed)


def tokenize(text: str) -> list[str]:
    """将英文文本转为小写单词列表。"""
    return TOKEN_PATTERN.findall(str(text).lower())


class TokenizedCorpus:
    """可重复遍历的分词语料，避免一次性保存全部分词结果。"""

    def __init__(self, texts: Sequence[str]) -> None:
        self.texts = texts

    def __iter__(self) -> Iterator[list[str]]:
        for text in self.texts:
            tokens = tokenize(text)
            if tokens:
                yield tokens


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, list[int]]]:
    """读取数据，并复用 Task 1 保存的数据划分。"""
    nyt = pd.read_csv(NYT_PATH, usecols=["text", "label"])
    nyt = nyt.dropna(subset=["text", "label"]).reset_index(drop=True)
    nyt["text"] = nyt["text"].astype(str)
    nyt["label"] = nyt["label"].astype(str)

    ag = pd.read_csv(AG_PATH, usecols=["text"])
    ag = ag.dropna(subset=["text"]).reset_index(drop=True)
    ag["text"] = ag["text"].astype(str)

    with SPLIT_PATH.open("r", encoding="utf-8") as file:
        split = json.load(file)

    all_indices = split["train"] + split["validation"] + split["test"]
    if len(all_indices) != len(nyt) or len(set(all_indices)) != len(nyt):
        raise ValueError("Task 1 的数据划分与当前 NYT 数据不一致。")
    return nyt, ag, split


def train_word2vec(texts: Sequence[str]) -> Word2Vec:
    """在给定语料上训练题目要求的 100 维 Word2Vec 模型。"""
    corpus = TokenizedCorpus(texts)
    model = Word2Vec(
        vector_size=VECTOR_SIZE,
        window=5,
        min_count=2,
        workers=1,
        sg=1,
        seed=SEED,
    )
    model.build_vocab(corpus)
    model.train(corpus, total_examples=model.corpus_count, epochs=5)
    return model


def vectorize_documents(
    texts: Iterable[str], keyed_vectors
) -> tuple[np.ndarray, dict[str, float | int]]:
    """对文档内所有有效词向量取平均，生成 100 维文档向量。"""
    texts = list(texts)
    matrix = np.zeros((len(texts), VECTOR_SIZE), dtype=np.float32)
    total_tokens = 0
    covered_tokens = 0
    zero_documents = 0

    for row, text in enumerate(texts):
        tokens = tokenize(text)
        total_tokens += len(tokens)
        vectors = [keyed_vectors[token] for token in tokens if token in keyed_vectors]
        covered_tokens += len(vectors)
        if vectors:
            matrix[row] = np.mean(vectors, axis=0)
        else:
            zero_documents += 1

    statistics: dict[str, float | int] = {
        "total_tokens": total_tokens,
        "covered_tokens": covered_tokens,
        "token_coverage": covered_tokens / total_tokens if total_tokens else 0.0,
        "zero_vector_documents": zero_documents,
    }
    return matrix, statistics


def evaluate(
    experiment: str,
    keyed_vectors,
    nyt: pd.DataFrame,
    split: dict[str, list[int]],
) -> tuple[dict, pd.DataFrame]:
    """生成三份文档向量、训练逻辑回归，并在测试集上评价。"""
    train_idx = np.asarray(split["train"])
    val_idx = np.asarray(split["validation"])
    test_idx = np.asarray(split["test"])
    texts = nyt["text"].to_numpy()
    labels = nyt["label"].to_numpy()

    x_train, train_stats = vectorize_documents(texts[train_idx], keyed_vectors)
    x_val, val_stats = vectorize_documents(texts[val_idx], keyed_vectors)
    x_test, test_stats = vectorize_documents(texts[test_idx], keyed_vectors)

    classifier = LogisticRegression(
        max_iter=2000,
        random_state=SEED,
        solver="lbfgs",
    )
    classifier.fit(x_train, labels[train_idx])
    predictions = classifier.predict(x_test)

    accuracy = accuracy_score(labels[test_idx], predictions)
    macro_f1 = f1_score(labels[test_idx], predictions, average="macro")
    report = classification_report(
        labels[test_idx], predictions, output_dict=True, zero_division=0
    )
    result = {
        "representation": experiment,
        "vector_size": VECTOR_SIZE,
        "classifier": "LogisticRegression",
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "coverage": {
            "train": train_stats,
            "validation": val_stats,
            "test": test_stats,
        },
        "classification_report": report,
    }
    prediction_table = pd.DataFrame(
        {
            "index": test_idx,
            "true_label": labels[test_idx],
            "predicted_label": predictions,
        }
    )

    print(f"\n{experiment}: Accuracy={accuracy:.6f}, Macro-F1={macro_f1:.6f}")
    print(f"Test token coverage={test_stats['token_coverage']:.4%}")
    print(classification_report(labels[test_idx], predictions, digits=4, zero_division=0))
    return result, prediction_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行 Task 2 的三组词向量实验。")
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=["glove", "ag_word2vec", "nyt_word2vec"],
        default=["glove", "ag_word2vec", "nyt_word2vec"],
        help="指定需要运行的实验，默认运行全部实验。",
    )
    return parser.parse_args()


def main() -> None:
    set_seed()
    args = parse_args()
    OUT_DIR.mkdir(exist_ok=True)
    nyt, ag, split = load_inputs()
    results_path = OUT_DIR / "results.json"
    if results_path.exists():
        with results_path.open("r", encoding="utf-8") as file:
            results = json.load(file)
    else:
        results = {}

    for experiment in args.experiments:
        if experiment == "glove":
            print("正在加载 glove-wiki-gigaword-100（GloVe 6B, 100维）……")
            vectors = api.load("glove-wiki-gigaword-100")
        elif experiment == "ag_word2vec":
            print("正在使用 AG News 文本训练 100 维 Word2Vec……")
            vectors = train_word2vec(ag["text"].to_numpy()).wv
        else:
            print("正在使用 NYT Training Set 文本训练 100 维 Word2Vec……")
            train_texts = nyt.loc[split["train"], "text"].to_numpy()
            vectors = train_word2vec(train_texts).wv

        result, predictions = evaluate(experiment, vectors, nyt, split)
        results[experiment] = result
        predictions.to_csv(OUT_DIR / f"{experiment}_predictions.csv", index=False)
        with results_path.open("w", encoding="utf-8") as file:
            json.dump(results, file, ensure_ascii=False, indent=2)

    print(f"\n实验结果已保存到：{results_path}")


if __name__ == "__main__":
    main()
