"""
medical_qa.py
================

This module implements a simple question–answering system for
medical queries.  It uses classical natural language processing
techniques to train a retrieval‑based model on a set of question and
answer pairs.  The core idea is to embed the questions using a TF–IDF
vectorizer and then perform nearest‑neighbor search at inference time.

The dataset provided (``mle_screening_dataset.csv``) contains 16,406
question/answer pairs covering a variety of medical topics.  Questions
are often repeated with slightly different phrasings and answers.  To
ensure reproducibility and ease of use, the module does not require
any external downloads and relies entirely on ``scikit‑learn`` and
``pandas``.

Usage
-----

Run this module as a script to train the model, evaluate it on a
held‑out test set, and display example interactions.  For example::

    python medical_qa.py

The script will:

* Load and preprocess the dataset.
* Split the data into training, validation and test partitions.
* Fit a TF–IDF vectorizer on the training questions.
* Build a nearest neighbor index on the question vectors.
* Evaluate the system on the test set using a token‑based F1 score.
* Print three example interactions demonstrating the QA behavior.

Limitations
-----------

The retrieval‑based approach is simple and interpretable but has
limitations:

* It can only return answers that already exist in the dataset.  If a
  query relates to a novel disease or concept not covered in the
  training data, the system may return an unrelated answer.
* Answer quality is evaluated using token overlap rather than
  semantic similarity.  This may penalize correct but rephrased
  answers.
* The model does not handle spelling mistakes or highly
  conversational queries well.  Incorporating more advanced
  embeddings (e.g., word2vec or transformer encoders) could
  improve robustness.

Nevertheless, the current implementation provides a solid baseline
demonstrating how to build a QA system from scratch with
interpretable components.
"""

import argparse
import os
import re
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import f1_score


def normalize_text(text: str) -> str:
    """Simple text normalization: lowercase and remove non‑alphanumeric
    characters except whitespace.

    Parameters
    ----------
    text : str
        Input text to normalize.

    Returns
    -------
    str
        Normalized text.
    """
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation and numbers, keep letters and spaces
    text = re.sub(r"[^a-z\s]", " ", text)
    # Replace multiple spaces with a single space
    text = re.sub(r"\s+", " ", text).strip()
    return text


@dataclass
class MedicalQASystem:
    """Retrieval‑based medical question–answering system.

    This class encapsulates the data structures and methods required to
    train, query and evaluate a simple TF–IDF + nearest neighbor model
    for answering medical questions.
    """

    dataset_path: str
    test_size: float = 0.1
    val_size: float = 0.1
    random_state: int = 42
    max_features: int = 50000
    vectorizer: TfidfVectorizer = field(init=False)
    nn_model: NearestNeighbors = field(init=False)
    train_df: pd.DataFrame = field(init=False)
    val_df: pd.DataFrame = field(init=False)
    test_df: pd.DataFrame = field(init=False)
    train_vectors: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        # Load and preprocess data
        self._load_and_split_data()
        # Initialize TF–IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            stop_words="english",
            analyzer="word",
            ngram_range=(1, 2),
        )
        # Fit vectorizer on training questions
        self.train_vectors = self.vectorizer.fit_transform(
            self.train_df["clean_question"]
        )
        # Fit nearest neighbor model
        self.nn_model = NearestNeighbors(
            n_neighbors=1, metric="cosine"
        )
        self.nn_model.fit(self.train_vectors)

    def _load_and_split_data(self) -> None:
        """Load dataset from CSV and split into train, validation and test
        sets.  Also create a cleaned version of the questions for
        vectorization.
        """
        df = pd.read_csv(self.dataset_path)
        # Remove rows with missing values
        df = df.dropna(subset=["question", "answer"])
        # Create cleaned question column
        df["clean_question"] = df["question"].apply(normalize_text)
        # First split off test
        # Perform random split into train/val/test.  We avoid stratifying
        # by question because most questions occur only once in the
        # dataset.  Stratification would fail if any class has fewer
        # than two examples.
        train_val_df, self.test_df = train_test_split(
            df,
            test_size=self.test_size,
            random_state=self.random_state,
            shuffle=True,
        )
        # Now split training into train and validation
        train_size = 1.0 - self.val_size / (1.0 - self.test_size)
        self.train_df, self.val_df = train_test_split(
            train_val_df,
            test_size=1 - train_size,
            random_state=self.random_state,
            shuffle=True,
        )

    def query(self, question: str, k: int = 1) -> List[Tuple[str, float]]:
        """Return top ``k`` answers for a given question.

        Parameters
        ----------
        question : str
            User query.
        k : int
            Number of answers to return.

        Returns
        -------
        List[Tuple[str, float]]
            A list of tuples containing the answer and its similarity
            score (1 ‑ cosine distance).  The list is sorted in
            descending order of similarity.
        """
        clean_q = normalize_text(question)
        vec = self.vectorizer.transform([clean_q])
        # nearest neighbors returns (distances, indices)
        distances, indices = self.nn_model.kneighbors(vec, n_neighbors=k)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            answer = self.train_df.iloc[idx]["answer"]
            score = 1.0 - dist
            results.append((answer, score))
        return results

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text by splitting on whitespace after normalization.
        """
        return normalize_text(text).split()

    def evaluate(self) -> float:
        """Evaluate the QA system on the test set using macro F1 score.

        For each test question, we retrieve the top answer and compute
        the token‑based F1 score between the predicted answer and the
        ground truth answer.  The macro average F1 across the test
        instances is returned.

        Returns
        -------
        float
            The mean F1 score across all test examples.
        """
        f1_scores: List[float] = []
        for _, row in self.test_df.iterrows():
            q = row["question"]
            true_answer = row["answer"]
            pred_answer, _score = self.query(q)[0]
            f1_scores.append(self._answer_f1(true_answer, pred_answer))
        return float(np.mean(f1_scores))

    def _answer_f1(self, gold: str, pred: str) -> float:
        """Compute token‑based F1 between two strings."""
        gold_tokens = self._tokenize(gold)
        pred_tokens = self._tokenize(pred)
        if not gold_tokens or not pred_tokens:
            return 0.0
        common = set(gold_tokens) & set(pred_tokens)
        # precision = |common| / |pred_tokens|
        precision = len(common) / len(pred_tokens)
        # recall = |common| / |gold_tokens|
        recall = len(common) / len(gold_tokens)
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate the medical QA system.")
    parser.add_argument(
        "--data", default="/home/oai/share/mle_screening_dataset.csv", help="Path to CSV dataset."
    )
    args = parser.parse_args()
    system = MedicalQASystem(dataset_path=args.data)
    print("\nEvaluating model on test set...")
    mean_f1 = system.evaluate()
    print(f"Mean token‑based F1 score: {mean_f1:.4f}\n")
    # Example interactions
    print("Example interactions:\n")
    examples = [
        "What are the symptoms of diabetes?",
        "How can I prevent glaucoma?",
        "What causes hypertension?",
    ]
    for example in examples:
        answers = system.query(example)
        print(f"Q: {example}")
        print(f"A: {answers[0][0]}\n")


if __name__ == "__main__":
    main()