import os
from abc import ABC, abstractmethod
import time
import plotly.express as px


class MetricsCollector:
    def __init__(self, dataset, n_specific_topics, experiment_id=None, save_path='./metrics/'):
        """

        :param dataset: dataset name
        :param n_specific_topics: number of specific (main) topics
        :param experiment_id: unique id of the experiment, by default time.time()
        :param save_path: folder where to store the results of experiments, be default ./metrics/
        """
        if not experiment_id:
            experiment_id = str(int(time.time()))
        self.save_path = os.path.join(save_path)
        self.save_fname = f'{experiment_id}_{dataset}_{n_specific_topics}.csv'
        self.dict_of_population = {}
        self.mutation_changes = []
        self.crossover_changes = []

    def save_fitness(self, generation: str, params: list, fitness: float):
        """

        :param generation: id of the generation
        :param params: hyperparameters of the individ
        :param fitness: fitness value
        :return:
        """
        if generation in self.dict_of_population:
            self.dict_of_population[f'gen_{generation}']['fitness'].append(fitness)
            self.dict_of_population[f'gen_{generation}']['params'].append(params)
        else:
            self.dict_of_population[f'gen_{generation}'] = {'fitness': [fitness], 'params': [params]}


    def write_metrics_to_file(self):
        os.makedirs(self.save_path, exist_ok=True)
        for gen in self.dict_of_population:
            fitness_vals = self.dict_of_population[gen]['fitness']

        # TODO: write save procedure
        raise NotImplementedError

    def visualise_trace(self):
        df = px.data.gapminder().query("country=='Canada'")
        fig = px.line(df, x="year", y="lifeExp", title='Life expectancy in Canada')
        fig.show()
