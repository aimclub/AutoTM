"""Metrics utilities for gensim_lda baseline."""

from typing import List, Optional, Tuple

import numpy as np
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel, LdaModel


try:  # pragma: no cover - fallback only when autotm is unavailable
    from autotm.fitness.external_scores import switchp  # type: ignore
except Exception:  # pragma: no cover - keep local fallback identical to AutoTM
    def switchp(phi, texts):
        words = phi.index.to_list()
        max_topic_word_dist = np.argmax(phi.to_numpy(), axis=1)
        max_topic_word_dist = dict(zip(words, max_topic_word_dist))
        switchp_scores = []
        for text in texts:
            mapped_text = [
                max_topic_word_dist[word]
                for word in text.split()
                if word in max_topic_word_dist
            ]
            if len(mapped_text) <= 1:
                switchp_scores.append(0.0)
                continue
            switches = (np.diff(mapped_text) != 0).sum()
            switchp_scores.append(switches / (len(mapped_text) - 1))

        return switchp_scores


def topics_top_words(model: LdaModel, topn: int) -> List[List[str]]:
    """Return the top ``topn`` words for each topic."""

    return [[word for word, _ in model.show_topic(topic_id, topn=topn)] for topic_id in range(model.num_topics)]


def compute_metrics(
    model: LdaModel,
    tokenized: List[List[str]],
    dictionary: Dictionary,
    *,
    topn: int = 25,
) -> Tuple[float, float, float]:
    """Compute AutoTM-compatible fitness, coherence and NPMI metrics."""

    topics_words = topics_top_words(model, topn)

    cm_cv = CoherenceModel(topics=topics_words, texts=tokenized, dictionary=dictionary, coherence="c_v")
    cv_per_topic = np.asarray(cm_cv.get_coherence_per_topic(), dtype=np.float64)

    dictionary_npmi = Dictionary(tokenized)
    cm_npmi = CoherenceModel(topics=topics_words, texts=tokenized, dictionary=dictionary_npmi, coherence="c_npmi")
    npmi_per_topic = np.asarray(cm_npmi.get_coherence_per_topic(), dtype=np.float64)

    fitness = float(np.mean(cv_per_topic) + np.min(cv_per_topic))
    coherence_25 = float(np.mean(cv_per_topic))
    npmi_25 = float(np.mean(npmi_per_topic))

    return fitness, coherence_25, npmi_25


def compute_switchp(model: LdaModel, dictionary: Dictionary, tokenized: List[List[str]]) -> Optional[float]:
    """Compute the SwitchP topic metric when possible."""

    try:
        topics = model.get_topics()
        words = [dictionary[idx] for idx in range(len(dictionary))]
        import pandas as _pd

        phi_df = _pd.DataFrame(topics.T, index=words, columns=[f"topic_{i}" for i in range(model.num_topics)])
        texts_raw = [" ".join(tokens) for tokens in tokenized]
        sp_scores = switchp(phi_df, texts_raw)
        sp_arr = np.asarray([score for score in sp_scores if score is not None and np.isfinite(score)], dtype=np.float64)
        return float(np.mean(sp_arr)) if sp_arr.size > 0 else None
    except Exception:
        return None


def doc_topic_matrix(model: LdaModel, corpus) -> np.ndarray:
    """Construct the dense document-topic matrix for a fitted model."""

    n_topics = model.num_topics
    matrix = np.zeros((len(corpus), n_topics), dtype=np.float32)
    for doc_idx, bow in enumerate(corpus):
        for topic_id, prob in model.get_document_topics(bow, minimum_probability=0.0):
            matrix[doc_idx, topic_id] = prob
    return matrix


def knn_f1(doc_topic: np.ndarray, labels: np.ndarray, *, n_splits: int, k_neighbors: int, random_seed: int) -> float:
    """Compute weighted F1 via KNN cross-validation."""

    from sklearn.model_selection import StratifiedKFold
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import f1_score

    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    scores = []
    for train_idx, test_idx in splitter.split(doc_topic, labels):
        clf = KNeighborsClassifier(n_neighbors=k_neighbors, weights="distance")
        clf.fit(doc_topic[train_idx], labels[train_idx])
        predictions = clf.predict(doc_topic[test_idx])
        scores.append(f1_score(labels[test_idx], predictions, average="weighted"))

    return float(np.mean(scores))


__all__ = [
    "compute_metrics",
    "compute_switchp",
    "doc_topic_matrix",
    "knn_f1",
    "topics_top_words",
]


