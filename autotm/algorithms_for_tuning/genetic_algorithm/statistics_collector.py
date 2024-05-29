from autotm.algorithms_for_tuning.individuals import Individual


class StatisticsCollector:
    """
    This logger handles collection of statistics
    """

    def log_iteration(self, evaluations: int, best_fitness: float):
        """
        :param evaluations: the number of used evaluations
        :param best_fitness: the best fitness in the current iteration
        """
        pass

    def log_individual(self, individual: Individual):
        """
        :param individual: a new evaluated individual
        """
        pass
