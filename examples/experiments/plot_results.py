import colorsys
import hashlib
import os
import random
import re
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

plt.rcParams["legend.loc"] = 'lower right'
plt.rcParams['figure.dpi'] = 300
sns.set(style="whitegrid")


def replace_with_max(array):
    array = np.array(array)
    array = np.maximum.accumulate(array)
    return array.tolist()


def git_hash_to_color(git_hash):
    color_palette = [
        "#FFB347",  # Prima
        "#FF6961",  # Light Red
        "#77DD77",  # Light Green
        "#03C03C",  # Dark Green
        "#779ECB",  # Dark Blue
        "#AEA1FF",  # Lavender
        "#F49AC2",  # Pink Sherbert
        "#CB99C9"  # Thistle
    ]

    hash_object = hashlib.md5(git_hash.encode())
    hex_dig = hash_object.hexdigest()
    return color_palette[int(hex_dig, 16) % len(color_palette)]


def map_git_hash(h):
    if h.startswith("b'"):
        h = h[2:-1]
    if h == "bbe897ef7e3c6b14bf47cf10fa344da4c061f1a9":
        return "initial"
    elif h == "7b9cd5510d0995009d5056fe0d27e7d746702cdb" or h == "0ee071168e08242be2e6d3786a7a62bb887c47f3":
        return "fix calculations"
    elif h == "fe16a7cd7ae77591f8f51ca8f960b5495118958d":
        return "cache dataset"
    elif h == "f764e22f967f6ea2e5447793e7f0751de1b2f98c":
        return "graph GA"
    elif h == "79c1fe4254e108107ecf89f5b50b1bf25c26da0a":
        return "fixed with all fixes"
    else:
        return h


def load_file(log_file, datasets):
    with open(log_file, 'r') as file:
        for line in file:
            parts = re.split(',(?![^\[]*\])', line.strip())
            git_hash = map_git_hash(parts[0])
            dataset = parts[1]
            use_pipeline = "True" == parts[2]
            iterations = list(map(int, re.findall(r'\d+', parts[3])))
            fitness = list(map(float, re.findall(r'\d+(\.\d+)?', parts[4])))
            fitness = replace_with_max(fitness)
            datasets[dataset].append((use_pipeline, iterations, fitness))


def transform_to_invocations(iterations, fitness):
    invocations = list(range(iterations[0], iterations[-1] + 1))
    new_fitness = [0] * len(invocations)
    min_inv = iterations[0]
    for i in range(len(iterations) - 1):
        inv = iterations[i]
        next_inv = iterations[i + 1]
        for j in range(inv - min_inv, next_inv - min_inv):
            new_fitness[j] = fitness[i]
    new_fitness[-1] = fitness[-1]
    cut_length = min(len(invocations), 150 - min_inv + 1)
    return invocations[:cut_length], new_fitness[:cut_length]


def plot_all():
    files_in_directory = os.listdir()
    log_files = [file for file in files_in_directory if re.match(r"log-\d{6}-\d{6}\.txt", file)]

    datasets = defaultdict(list)
    for log_file in log_files:
        load_file(log_file, datasets)

    final_results = []
    for dataset in datasets:
        use_pipeline = []
        invocations = []
        fitness = []
        for p, i, f in datasets[dataset]:
            i, f = transform_to_invocations(i, f)
            invocations += i
            fitness += f
            use_pipeline += [p] * len(i)
            final_results.append((dataset, p, f[-1]))

        df = pd.DataFrame(data={'invocations': invocations, 'fitness': fitness,
                                'use_pipeline': use_pipeline})
        sns.lineplot(data=df, x='invocations', y='fitness', hue='use_pipeline', ci=90, markers=True)
        plt.title(dataset)
        plt.savefig(f"{datetime.now().strftime('%y%m%d-%H%M%S')}_{dataset}.png")
        plt.show()

    ds = []
    ps = []
    fs = []
    for dataset, use_pipeline, final_fitness in final_results:
        ds.append(dataset.replace("_sample", ""))
        ps.append(use_pipeline)
        fs.append(final_fitness)
    df_final = pd.DataFrame(data={'dataset': ds, 'use_pipeline': ps, 'fitness': fs})
    print(df_final.groupby(["dataset", "use_pipeline"])["fitness"].mean())
    print(df_final.groupby(["dataset", "use_pipeline"])["fitness"].std())
    sns.boxplot(data=df_final, x='dataset', y='fitness', hue='use_pipeline')
    plt.savefig(f"{datetime.now().strftime('%y%m%d-%H%M%S')}_final_results.png")
    plt.show()


if __name__ == "__main__":
    plot_all()
