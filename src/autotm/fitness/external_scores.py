import numpy as np


# additional scores to calculate


# Topic Significance
def kl_divergence(p, q):
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))


# Uniform Distribution Over Words (W-Uniform)
def ts_uniform(topic_word_dist):
    n_words = topic_word_dist.shape[0]
    w_uniform = np.ones(n_words) / n_words
    uniform_distances_kl = [kl_divergence(p, w_uniform) for p in topic_word_dist.T]
    return uniform_distances_kl


# Vacuous Semantic Distribution (W-Vacuous)
def ts_vacuous(doc_topic_dist, topic_word_dist, total_tokens):
    n_words = topic_word_dist.shape[0]
    #     n_tokens = np.sum([len(text) for text in texts])
    p_k = np.sum(doc_topic_dist, axis=1) / total_tokens
    w_vacauous = np.sum(topic_word_dist * np.tile(p_k, (n_words, 1)), axis=1)
    vacauous_distances_kl = [kl_divergence(p, w_vacauous) for p in topic_word_dist.T]
    return vacauous_distances_kl


# Background Distribution (D-BGround)
def ts_bground(doc_topic_dist):
    n_documents = doc_topic_dist.shape[1]
    d_bground = np.ones(n_documents) / n_documents
    d_bground_distances_kl = [kl_divergence(p.T, d_bground) for p in doc_topic_dist]
    return d_bground_distances_kl


# SwitchP
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
        switches = (np.diff(mapped_text) != 0).sum()
        switchp_scores.append(switches / (len(mapped_text) - 1))

    return switchp_scores
