import click
import numpy as np
from scipy.optimize import differential_evolution


def type_check(res):
    res = list(res)
    for i in [1, 4, 7, 10, 11]:
        res[i] = int(res[i])
    return res


@click.command(context_settings=dict(allow_extra_args=True))
@click.option('--dataset', help='dataset name in the config')
@click.option('--strategy', default='best1bin', help='strategy of the algorithm')  # colony size
@click.option('--maxiter', default=None, help='maximum number of generations to evolve')
@click.option('--popsize', default=10, help='population size')
@click.option('--tol', default=None, help='relative tolerance for convergence')
@click.option('--mutation', default=None, help='mutation constant')
@click.option('--recombination', default=None, help='recombinaiton constant')
@click.option('--init', default=None, help='type of population initialization')
@click.option('--atol', default=None, help='absolute tolerance for convergence')
@click.option('--log-file', default="/var/log/tm-alg.log",
              help='a log file to write logs of the algorithm execution to')
def run_algorithm():

    raise NotImplementedError


def BigartmOptimizer(x):
    params = x
    print('PARAMS: ', params)
    t = Topic_model(experiments_path, S=S)

    params = type_check(params)
    t.init_model(params)
    t.train()

    try:
        scores = get_last_avg_vals(t.model, S)
        print(scores)
    except:
        print('NO SCORES SHALL PASS!')

    # Additional penalty for creating unmeaningfull main topics
    #     theta = t.model.get_theta()
    #     theta_trans = theta.T[['main{}'.format(i) for i in range(S)]]
    #     theta_trans_vals = theta_trans.max(axis=1)
    #     fine_elems_coeff = len(np.where(np.array(theta_trans_vals)>=0.2)[0])/theta_trans.shape[0]
    try:
        fitness = t.get_avg_coherence_score(for_individ_fitness=True)  # *fine_elems_coeff
    except:
        fitness = 0
    if np.isnan(fitness):
        fitness = 0
    print()
    print('CURRENT FITNESS: {}'.format(fitness))
    print()
    return -fitness


if __name__ == "__main__":
    run_algorithm()
