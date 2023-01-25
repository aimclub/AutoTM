import os
import numpy as np
from abc import ABC, abstractmethod
import time

import pandas as pd
import plotly.express as px
from scipy.spatial import distance

GENERATION_COL = 'generation'
FITNESS_COL = 'fitness'
FITNESS_DIFF_COL = 'fitness_diff'
PARAMS_DIST_COL = 'params_dist'


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
        self.save_fname = f'{experiment_id}_{dataset}_{n_specific_topics}'
        self.dict_of_population = {}
        self.mutation_changes = {}
        self.crossover_changes = {}
        self.num_generations = 0
        self.metric_df = None
        self.mutation_df = None

    def save_mutation(self, generation: int, original_params: list, mutated_params: list, original_fitness: float,
                      mutated_fitness: float):
        # excluding mutation parameters
        eucledian_distance = distance.euclidean(original_params[:11] + [original_params[15]],
                                                mutated_params[:11] + [mutated_params[15]])
        fitness_diff = mutated_fitness - original_fitness

        if f'gen_{generation}' in self.mutation_changes:
            self.mutation_changes[f'gen_{generation}']['params_dist'].append(eucledian_distance)
            self.mutation_changes[f'gen_{generation}']['fitness_diff'].append(fitness_diff)
            self.mutation_changes[f'gen_{generation}']['original_params'].append(
                original_params[:11] + [original_params[15]])
            self.mutation_changes[f'gen_{generation}']['mutated_params'].append(
                mutated_params[:11] + [mutated_params[15]])
        else:
            self.mutation_changes[f'gen_{generation}'] = {'params_dist': [eucledian_distance],
                                                          'fitness_diff': [fitness_diff],
                                                          'original_params': [
                                                              original_params[:11] + [original_params[15]]],
                                                          'mutated_params': [
                                                              mutated_params[:11] + [mutated_params[15]]]}

    def save_crossover(self, generation: int, parent_1: list, parent_2: list, child_1: list, parent_1_fitness: float,
                       parent_2_fitness: float, child_1_fitness: float, child_2: list = None,
                       child_2_fitness: list = None):
        """

        :param generation:
        :param parent_1:
        :param parent_2:
        :param child_1:
        :param parent_1_fitness:
        :param parent_2_fitness:
        :param child_1_fitness:
        :param child_2:
        :param child_2_fitness:
        :return:
        """
        if f'gen_{generation}' in self.crossover_changes:
            self.crossover_changes[f'gen_{generation}']['parent_1_params'].append(parent_1)
            self.crossover_changes[f'gen_{generation}']['parent_2_params'].append(parent_2)
            self.crossover_changes[f'gen_{generation}']['child_1_params'].append(child_1)
            self.crossover_changes[f'gen_{generation}']['parent_1_fitness'].append(parent_1_fitness)
            self.crossover_changes[f'gen_{generation}']['parent_2_fitness'].append(parent_2_fitness)
            self.crossover_changes[f'gen_{generation}']['child_1_fitness'].append(child_1_fitness)
        else:
            self.crossover_changes[f'gen_{generation}'] = {'parent_1_params': [parent_1],
                                                           'parent_2_params': [parent_2],
                                                           'child_1_params': [child_1],
                                                           'parent_1_fitness': [parent_1_fitness],
                                                           'parent_2_fitness': [parent_2_fitness],
                                                           'child_1_fitness': [child_1_fitness]}
            if child_2 is not None:
                self.crossover_changes[f'gen_{generation}']['child_2_params'] = [child_2]
                self.crossover_changes[f'gen_{generation}']['child_2_fitness'] = [child_2_fitness]

    def save_fitness(self, generation: int, params: list, fitness: float):
        """

        :param generation: id of the generation
        :param params: hyperparameters of the individ
        :param fitness: fitness value
        :return:
        """

        if generation > self.num_generations:
            self.num_generations = generation

        if f'gen_{generation}' in self.dict_of_population:
            self.dict_of_population[f'gen_{generation}']['fitness'].append(fitness)
            self.dict_of_population[f'gen_{generation}']['params'].append(params)
        else:
            self.dict_of_population[f'gen_{generation}'] = {'fitness': [[fitness]], 'params': [[params]]}

    def get_metric_df(self):
        if self.metric_df is not None:
            print('Metric df already exists')
        else:
            population_max = []
            for i in range(self.num_generations + 1):
                gen_value = np.max(self.dict_of_population[f'gen_{i}']['fitness'])
                if i == 0:
                    population_max = [gen_value]
                else:
                    population_max.append(max(population_max[i - 1], gen_value))
            self.metric_df = pd.DataFrame(list(zip([i for i in range(self.num_generations + 1)], population_max)),
                                          columns=[GENERATION_COL, FITNESS_COL])
        if self.mutation_df is not None:
            print('Mutation df already exists')
        else:
            dfs = []
            for gen in self.mutation_changes:
                cur_df_dict = {PARAMS_DIST_COL: self.mutation_changes[gen]['params_dist'],
                               FITNESS_DIFF_COL: self.mutation_changes[gen]['fitness_diff'],
                               'original_params': self.mutation_changes[gen]['original_params'],
                               'mutated_params': self.mutation_changes[gen]['mutated_params']}
                cur_df = pd.DataFrame(
                    cur_df_dict
                )
                cur_df[GENERATION_COL] = gen
                dfs.append(cur_df)
            self.mutation_df = pd.concat(dfs)

    def write_metrics_to_file(self):
        os.makedirs(self.save_path, exist_ok=True)
        self.metric_df.to_csv(os.path.join(self.save_path, f'{self.save_fname}_metric_{int(time.time())}.csv'))
        self.mutation_df.to_csv(os.path.join(self.save_path, f'{self.save_fname}_mutation_{int(time.time())}.csv'))

    def visualise_trace(self):
        self.get_metric_df()
        # traces vis
        graph_template = 'plotly_white'

        fig = px.line(self.metric_df, x=GENERATION_COL, y=FITNESS_COL, title='Score changes with generations',
                      template=graph_template)
        fig.show()

        # mutation diff vis
        fig = px.scatter(self.mutation_df, x=PARAMS_DIST_COL, y=FITNESS_DIFF_COL,
                         title='Effectiveness of mutation operation', template=graph_template)
        fig.show()

        # crossover diff vis

        # save params
        self.write_metrics_to_file()
