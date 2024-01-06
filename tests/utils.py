import pandas as pd
from numpy._typing import ArrayLike

from autotm.base import AutoTM


def check_predictions(autotm: AutoTM, df: pd.DataFrame, mixtures: ArrayLike):
    n_samples, n_samples_mixture = df.shape[0], mixtures.shape[0]
    n_topics, n_topics_mixture = len(autotm.topics), mixtures.shape[1]

    assert n_samples_mixture == n_samples
    assert n_topics_mixture == n_topics
    assert (~mixtures.isna()).all().all()
    assert (~mixtures.isnull()).all().all()
