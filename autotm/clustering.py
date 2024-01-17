from matplotlib import offsetbox
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import seaborn as sns
import pandas as pd
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import warnings
import copy


warnings.filterwarnings('ignore')


def cluster_phi(phi_df: pd.dataFrame, n_clusters=10, plot_img=True):
    _phi_df = copy.deepcopy(phi_df)
    y = _phi_df.index.values
    x = _phi_df.values
    standardized_x = StandardScaler().fit_transform(x)
    y_kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(standardized_x)

    if plot_img:
        tsne = TSNE(n_components=2).fit_transform(standardized_x)
        plt.scatter(tsne[:, 0], tsne[:, 1], c=y_kmeans.labels_, s=6, cmap='Spectral')
        plt.gca().set_aspect('equal', 'datalim')
        plt.colorbar(boundaries=np.arange(11) - 0.5).set_ticks(np.arange(10))
        plt.title('Scatterplot of lenta.ru data', fontsize=24)

    _phi_df['labels'] = y_kmeans.labels_
    return _phi_df


