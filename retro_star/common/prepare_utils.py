import pickle
import pandas as pd
import logging
# from mlp_retrosyn.mlp_inference import MLPModel
from retro_star.alg import molstar


def prepare_starting_molecules(filename):
    logging.info('Loading starting molecules from %s' % filename)

    if filename[-3:] == 'csv':
        starting_mols = set(list(pd.read_csv(filename)['mol']))
    else:
        assert filename[-3:] == 'pkl'
        with open(filename, 'rb') as f:
            starting_mols = pickle.load(f)

    logging.info('%d starting molecules loaded' % len(starting_mols))
    return starting_mols

def prepare_molstar_planner(one_step, value_fn, expansion_topk,
                            iterations, viz=False, viz_dir=None, starting_mols=None):
    expansion_handle = lambda x: one_step.run(x, topk=expansion_topk)

    if starting_mols is not None:
        plan_handle = lambda x, y=0: molstar(
            target_mol=x,
            target_mol_id=y,
            starting_mols=starting_mols,
            expand_fn=expansion_handle,
            value_fn=value_fn,
            iterations=iterations,
            viz=viz,
            viz_dir=viz_dir
        )
    else:
        plan_handle = lambda x, z, y=0,: molstar(
            target_mol=x,
            target_mol_id=y,
            starting_mols=z,
            expand_fn=expansion_handle,
            value_fn=value_fn,
            iterations=iterations,
            viz=viz,
            viz_dir=viz_dir
        )
    return plan_handle
