import copy
import logging
import random
from typing import List

import artm
from pydantic import BaseModel

from autotm.abstract_params import AbstractParams
from autotm.fitness.tm import TopicModel
from autotm.graph_ga import create_pipeline, crossover_pipelines, mutate_pipeline
from autotm.pipeline import Pipeline, Stage, StageType, Param, create_stage, IntRangeDistribution, \
    FloatRangeDistribution

PARAM_NAMES = [
    "val_decor",
    "var_n_0",
    "var_sm_0",
    "var_sm_1",
    "var_n_1",
    "var_sp_0",
    "var_sp_1",
    "var_n_2",
    "var_sp_2",
    "var_sp_3",
    "var_n_3",
    "B",
    "ext_mutation_prob",
    "ext_elem_mutation_prob",
    "ext_mutation_selector",
    "val_decor_2",
]


def type_check(res: List):
    res = list(res)
    for i in [1, 4, 7, 10, 11]:
        res[i] = int(res[i])
    return res


class FixedListParams(BaseModel, AbstractParams):
    params: List[float]

    @property
    def basic_topics(self):
        return int(self.params[11])

    @property
    def mutation_probability(self):
        return self.params[12]

    def to_pipeline_params(self) -> "PipelineParams":
        stages = []
        if self.params[1] > 0:
            stages.append(
                Stage(stage_type=DECORRELATION_STAGE_TYPE, values=[self.params[1], self.params[0], self.params[15]]))
        if self.params[4] > 0:
            stages.append(Stage(stage_type=SMOOTH_STAGE_TYPE, values=[self.params[4], self.params[2], self.params[3]]))
        if self.params[7] > 0:
            stages.append(Stage(stage_type=SPARSE_STAGE_TYPE, values=[self.params[7], self.params[5], self.params[6]]))
        if self.params[10] > 0:
            stages.append(Stage(stage_type=SPARSE_STAGE_TYPE, values=[self.params[10], self.params[8], self.params[9]]))

        required_params = Stage(stage_type=REQUIRED_STAGE_TYPE, values=self.params[11:15])
        return PipelineParams(pipeline=Pipeline(stages=stages, required_params=required_params))

    def validate_params(self) -> bool:
        stage = Stage(stage_type=FIXED_LIST_STAGE_TYPE, values=self.params)
        stage.clip_values()
        self.params = stage.values
        self.params = type_check(self.params)
        return True

    def make_params_dict(self):
        if len(self.params) > len(PARAM_NAMES):
            len_diff = len(self.params) - len(PARAM_NAMES)
            param_names = copy.deepcopy(PARAM_NAMES) + [
                f"unknown_param_#{i}" for i in range(len_diff)
            ]
        else:
            param_names = PARAM_NAMES

        return {name: p_val for name, p_val in zip(param_names, self.params)}

    def run_train(self, model: TopicModel):
        self.to_pipeline_params().run_train(model)

    def crossover(self, parent2: "AbstractParams", **kwargs) -> List["AbstractParams"]:
        assert isinstance(parent2, FixedListParams)
        from autotm.algorithms_for_tuning.genetic_algorithm.crossover import crossover
        crossover_fun = crossover(kwargs["crossover_type"])
        children = crossover_fun(copy.deepcopy(self.params), copy.deepcopy(parent2.params), **kwargs)
        return [FixedListParams(params=values) for values in children]

    def mutate(self, **kwargs) -> "AbstractParams":
        from autotm.algorithms_for_tuning.genetic_algorithm.mutation import mutation
        mutation_fun = mutation(kwargs["mutation_type"])
        params = copy.deepcopy(self.params)
        elem_mutation_prob = params[13]
        params = mutation_fun(params, elem_mutation_prob=elem_mutation_prob)

        # TODO: check this code
        for ix in [12, 13, 14]:
            if random.random() < elem_mutation_prob:
                params[ix] = META_PROBABILITY_DISTRIBUTION.create_value()
        return FixedListParams(params=params)

    def to_vector(self) -> List[float]:
        return self.params


# ext_mutation_prob, ext_elem_mutation_prob, ext_mutation_selector
BASIC_TOPICS_PARAM = Param(name="basic_topics_count", distribution=IntRangeDistribution(low=0, high=5))
META_PROBABILITY_DISTRIBUTION = FloatRangeDistribution(low=0, high=1)
MUTATION_PROBABILITY_PARAM = Param(name="ext_mutation_prob", distribution=META_PROBABILITY_DISTRIBUTION)
ELEMENT_MUTATION_PROBABILITY_PARAM = Param(name="ext_elem_mutation_prob", distribution=META_PROBABILITY_DISTRIBUTION)
MUTATION_SELECTOR_PARAM = Param(name="ext_mutation_selector", distribution=META_PROBABILITY_DISTRIBUTION)
REQUIRED_STAGE_TYPE = StageType(name="General", params=[
    BASIC_TOPICS_PARAM,
    MUTATION_PROBABILITY_PARAM,
    ELEMENT_MUTATION_PROBABILITY_PARAM,
    MUTATION_SELECTOR_PARAM])

ITERATIONS_COUNT_PARAM = Param(name="n", distribution=IntRangeDistribution(low=1, high=30))

DECORRELATION_PARAM_DISTRIBUTION = FloatRangeDistribution(low=0, high=1e5)
DECORRELATION_STAGE_TYPE = StageType(name="DecorrelatorPhiRegularizer", params=[
    ITERATIONS_COUNT_PARAM,
    Param(name="decorr", distribution=DECORRELATION_PARAM_DISTRIBUTION),
    Param(name="decorr_2", distribution=DECORRELATION_PARAM_DISTRIBUTION)])

SMOOTH_PARAM_DISTRIBUTION = FloatRangeDistribution(low=0, high=1e2)
SMOOTH_STAGE_TYPE = StageType(name="SmoothThetaRegularizer", params=[
    ITERATIONS_COUNT_PARAM,
    Param(name="SmoothPhi", distribution=SMOOTH_PARAM_DISTRIBUTION),
    Param(name="SmoothTheta", distribution=SMOOTH_PARAM_DISTRIBUTION)])

SPARSE_PARAM_DISTRIBUTION = FloatRangeDistribution(low=-1e3, high=1e3)
SPARSE_STAGE_TYPE = StageType(name="SparseThetaRegularizer", params=[
    ITERATIONS_COUNT_PARAM,
    Param(name="SparsePhi", distribution=SPARSE_PARAM_DISTRIBUTION),
    Param(name="SparseTheta", distribution=SPARSE_PARAM_DISTRIBUTION)])

STAGE_TYPES = [DECORRELATION_STAGE_TYPE, SMOOTH_STAGE_TYPE, SPARSE_STAGE_TYPE]

ITERATIONS_FIXED_DISTRIBUTION = IntRangeDistribution(low=0, high=30)
FIXED_LIST_STAGE_TYPE = StageType(name="FixedListStage", params=[
    Param(name="val_decor", distribution=DECORRELATION_PARAM_DISTRIBUTION),  # 0
    Param(name="var_n_0", distribution=ITERATIONS_FIXED_DISTRIBUTION),  # 1
    Param(name="var_sm_0", distribution=SMOOTH_PARAM_DISTRIBUTION),  # 2
    Param(name="var_sm_1", distribution=SMOOTH_PARAM_DISTRIBUTION),  # 3
    Param(name="var_n_1", distribution=ITERATIONS_FIXED_DISTRIBUTION),  # 4
    Param(name="var_sp_0", distribution=SPARSE_PARAM_DISTRIBUTION),  # 5
    Param(name="var_sp_1", distribution=SPARSE_PARAM_DISTRIBUTION),  # 6
    Param(name="var_n_2", distribution=ITERATIONS_FIXED_DISTRIBUTION),  # 7
    Param(name="var_sp_2", distribution=SPARSE_PARAM_DISTRIBUTION),  # 8
    Param(name="var_sp_3", distribution=SPARSE_PARAM_DISTRIBUTION),  # 9
    Param(name="var_n_3", distribution=ITERATIONS_FIXED_DISTRIBUTION),  # 10
    Param(name="B", distribution=BASIC_TOPICS_PARAM.distribution),  # 11
    Param(name="ext_mutation_prob", distribution=META_PROBABILITY_DISTRIBUTION),  # 12
    Param(name="ext_elem_mutation_prob", distribution=META_PROBABILITY_DISTRIBUTION),  # 13
    Param(name="ext_mutation_selector", distribution=META_PROBABILITY_DISTRIBUTION),  # 14
    Param(name="val_decor_2", distribution=DECORRELATION_PARAM_DISTRIBUTION),  # 15
])


class PipelineParams(BaseModel, AbstractParams):
    pipeline: Pipeline

    @property
    def basic_topics(self):
        return int(self._get_required_param_value(BASIC_TOPICS_PARAM))

    @property
    def mutation_probability(self):
        return self._get_required_param_value(MUTATION_PROBABILITY_PARAM)

    def _get_required_param_value(self, param):
        i = self.pipeline.required_params.stage_type.params.index(param)
        return self.pipeline.required_params.values[i]

    def validate_params(self) -> bool:
        if len(self.pipeline.stages) == 0 or len(self.pipeline.stages) > 10:
            return False

        # parity with Fixed params
        if sum(stage.values[0] for stage in self.pipeline.stages) > ITERATIONS_COUNT_PARAM.distribution.high * 4:
            return False

        for stage in self.pipeline.stages:
            if stage.stage_type == SMOOTH_STAGE_TYPE and self.basic_topics == 0:
                return False

        for stage in self.pipeline.stages + [self.pipeline.required_params]:
            stage.clip_values()

        return True

    def run_train(self, model: TopicModel):
        for stage in self.pipeline.stages:
            if stage.stage_type == DECORRELATION_STAGE_TYPE:
                n, decorr, decorr_2 = stage.values
                model.model.regularizers.add(artm.DecorrelatorPhiRegularizer(
                    name="decorr", topic_names=model.specific, tau=decorr), overwrite=True)
                model.model.regularizers.add(artm.DecorrelatorPhiRegularizer(
                    name="decorr_2", topic_names=model.back, tau=decorr_2), overwrite=True)
                model.do_fit(n)
            elif stage.stage_type == SMOOTH_STAGE_TYPE:
                n, phi, theta = stage.values
                model.model.regularizers.add(artm.SmoothSparseThetaRegularizer(
                    name="SmoothPhi", topic_names=model.back, tau=phi), overwrite=True)
                model.model.regularizers.add(artm.SmoothSparseThetaRegularizer(
                    name="SmoothTheta", topic_names=model.back, tau=theta), overwrite=True)
                model.do_fit(n)
            elif stage.stage_type == SPARSE_STAGE_TYPE:
                n, phi, theta = stage.values
                model.model.regularizers.add(artm.SmoothSparseThetaRegularizer(
                    name="SparsePhi", topic_names=model.specific, tau=phi), overwrite=True)
                model.model.regularizers.add(artm.SmoothSparseThetaRegularizer(
                    name="SparseTheta", topic_names=model.specific, tau=theta), overwrite=True)
                model.do_fit(n)
            else:
                raise ValueError(f"Unknown stage type {stage.stage_type.name}")

            if model.check_early_stop():
                logging.info("Early stopping is triggered")
                return

    def make_params_dict(self):
        return ({f"0_{self.pipeline.required_params.stage_type.name}_{param.name}": value
                 for value, param in zip(self.pipeline.required_params.values,
                                         self.pipeline.required_params.stage_type.params)} |
                {f"{i + 1}_{stage.stage_type.name}_{param.name}": value
                 for i, stage in enumerate(self.pipeline.stages)
                 for value, param in zip(stage.values, stage.stage_type.params)})

    def crossover(self, parent2: "AbstractParams", **kwargs) -> List["AbstractParams"]:
        assert isinstance(parent2, PipelineParams)
        pipelines = crossover_pipelines(copy.deepcopy(self.pipeline), copy.deepcopy(parent2.pipeline))
        return [PipelineParams(pipeline=pipeline) for pipeline in pipelines]

    def mutate(self, **kwargs) -> "AbstractParams":
        pipeline = copy.deepcopy(self.pipeline)
        stage_mutation_probability = self._get_required_param_value(ELEMENT_MUTATION_PROBABILITY_PARAM)
        pipeline = mutate_pipeline(pipeline, STAGE_TYPES, stage_mutation_probability)
        return PipelineParams(pipeline=pipeline)

    def to_vector(self) -> List[float]:
        pipeline = self.pipeline
        max_stages_of_type = 10
        result = []
        for type_index, stage_type in enumerate(STAGE_TYPES):
            stages = iterations_of_type(pipeline.stages, stage_type.name)
            for i in range(max_stages_of_type):
                if i < len(stages):
                    result += stages[i].values
                else:
                    result += [0.] * len(stage_type.params)
        result.append(self.basic_topics)
        result.append(len(pipeline.stages))
        return result


def create_individual(base_model: bool, use_pipeline: bool) -> AbstractParams:
    while True:
        if use_pipeline:
            if base_model:
                params = PipelineParams(pipeline=Pipeline(stages=[create_stage(DECORRELATION_STAGE_TYPE)],
                                                          required_params=create_stage(REQUIRED_STAGE_TYPE)))
            else:
                pipeline = create_pipeline(STAGE_TYPES, lambda: random.randint(2, 4), REQUIRED_STAGE_TYPE)
                params = PipelineParams(pipeline=pipeline)
        else:
            stage = create_stage(FIXED_LIST_STAGE_TYPE)
            values = stage.values
            if base_model:
                for i in [0, 4, 7, 10, 11, 15]:
                    values[i] = 0
            params = FixedListParams(params=values)
        if params.validate_params():
            return params


def iterations_of_type(stages, stage_type):
    return [stage for stage in stages if stage.stage_type.name == stage_type]
