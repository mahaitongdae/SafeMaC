# The environment for this file is a ~/work/rl
import argparse
import errno
import os
import pickle
import warnings

import matplotlib.pyplot as plt
import torch
import yaml
import numpy as np
import pandas as pd

from utils.environement import GridWorld
from utils.ground_truth import GroundTruth
from utils.helper import submodular_optimization, idxfromloc
from utils.initializer import get_players_initialized



def train(args):

    # 1) Load the config file
    with open(workspace + "/params/" + args.param + ".yaml") as file:
        params = yaml.load(file, Loader=yaml.FullLoader)
    print(params)

    env_load_path = (
        workspace
        + "/experiments/"
        + params["experiment"]["folder"]
        + "/env_"
        + str(args.env)
        + "/"
    )
    save_path = env_load_path + "/" + args.param + "/"

    if not os.path.exists(save_path):
        try:
            os.makedirs(save_path)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

    # 3) Setup the environement
    env = GridWorld(
        env_params=params["env"], common_params=params["common"], env_dir=env_load_path
    )

    grid_V = env.grid_V
    # If optimistic set is intersection of 2 common graph, then they take a joint step (solve greedy algorithm combined)
    # If pessimistic set is combined, new pessimistic set is union of sets, and agent can travel in union
    init_safe = env.get_safe_init()
    print("initialized location", init_safe)

    # 3.1) Compute optimal location using true function to be used in regret computation
    opt = GroundTruth(env, params)
    opt.compute_optimal_location_by_expansion()
    opt.compute_normalization_factor()

    # while running on cluster in parallel sometimes a location in not created if asked by multiple processes
    os.makedirs(save_path, exist_ok=True)

    # use df to store data.
    data = {
        'current_coverage': [],
        # 'sum_max_sigma'     :[],
        'iter': [],
    }
    data.update({'idx_agent{}'.format(i): [] for i in range(params["env"]["n_players"])})




    # Start from some safe initial states
    train = {}
    train["Cx_X"] = init_safe["loc"]
    train["Cx_Y"] = env.get_multi_constraint_observation(train["Cx_X"])
    train["Fx_X"] = init_safe["loc"]
    train["Fx_Y"] = env.get_multi_density_observation(train["Fx_X"])
    players = get_players_initialized(train, params, grid_V)

    for it, player in enumerate(players):
        player.update_Fx_gp_with_current_data()
        player.update_graph(init_safe["idx"][it])
        player.save_posterior_normalization_const()  # agent.posterior_normalization_const, max of 2beta*sigma.
        player.initialize_location(init_safe["loc"][it])

    for i, player in enumerate(players):
        data['idx_agent{}'.format(i)].append(idxfromloc(player.grid_V, player.current_location))

    associate_dict = {}
    associate_dict[0] = []
    for idx in range(params["env"]["n_players"]):
        associate_dict[0].append(idx)

    pessi_associate_dict = {}
    pessi_associate_dict[0] = []
    for idx in range(params["env"]["n_players"]):
        pessi_associate_dict[0].append(idx)

    iter = 0
    doubling_target_iter = 0
    pt1 = None

    # compute coverage based on the initial location
    current_coverage = opt.compute_current_multiple_coverage(players, associate_dict)
    data.get('current_coverage').append(current_coverage)
    data.get('iter').append(0)

    # max_density_sigma = sum([player.get_max_sigma() for player in players]) # sigma is for objective Fx
    # data.get('sum_max_sigma').append(max_density_sigma)
    # print(iter, max_density_sigma)

    # 4) Solve the submodular problem and get a next point to go xi* in pessimistic safe set
    associate_dict, pessi_associate_dict, acq_density, M_dist = submodular_optimization(
        players, init_safe, params
    )
    '''
    haitong: in submodular_optimization, goal is set by agent.planned_dist_center & agent.planned_measured_loc.
                we keep both for compatible with previous visualization code.
    '''

    for player in players:
        player.planning(acq_density)

    while iter < params["algo"]["n_iter"]:
        '''
        haitong: main while loop.
        '''

        target_reached = []
        current_locations = []
        for i, player in enumerate(players):
            target_reached.append(player.update_current_location())
            current_locations.append(player.current_location)
            # for i, player in enumerate(players):
            data['idx_agent{}'.format(i)].append(idxfromloc(player.grid_V, player.current_location))

        # observation
        # current_locations = torch.from_numpy(np.array(current_locations))
        obs = env.get_multi_density_observation(current_locations)
        players[0].update_Fx_set(torch.stack(current_locations), torch.cat(obs))
        # haitong: wierd data dim from previous code.
        # haitong: Centralized algo currently so we only add to first agent then sync the FX model.


        if all(target_reached):
            if params["algo"]["use_doubling_trick"] and iter > doubling_target_iter:

                # only recal doubling trick if finished doubling trick last time
                current_samples = []
                for i, player in enumerate(players):
                    current_samples.append(data['idx_agent{}'.format(i)].count(idxfromloc(player.grid_V, player.current_location)))
                min_samples = min(current_samples)
                doubling_target_iter = iter + min_samples
                print('double until iter {}'.format(doubling_target_iter))

            if iter == doubling_target_iter or (not params["algo"]["use_doubling_trick"]):
                # if finish doubling trick this time, update GP and do planning.
                Fx_model = players[0].update_Fx()
                for i in range(1, len(players)):
                    players[i].Fx_model = Fx_model  # sync all Fx model.

                # Greedy algorithm to get path planning target.
                (
                    associate_dict,
                    pessi_associate_dict,
                    acq_density,
                    M_dist,
                ) = submodular_optimization(players, init_safe, params)

                # path planning.
                path_len = []
                for player in players:
                    acq_coverage = torch.stack(list(M_dist[0].values())).detach().numpy()
                    player.planning(acq_coverage) # todo: we planning according to single node reward or coverage reward.
                    path_len.append(len(player.path))
                print('max path len {}'.format(max(path_len)))

        iter += 1
        # max_density_sigma = sum(
        #     [player.max_density_sigma for player in players]
        # )
        # data.get('sum_max_sigma').append(max_density_sigma)
        # print(iter, max_density_sigma)

        current_coverage = opt.compute_current_multiple_coverage(players, associate_dict)
        data.get('current_coverage').append(current_coverage)

        data.get('iter').append(iter)
        print("Iter: {}, coverage value: {:.3f}".format(iter, current_coverage))

    df = pd.DataFrame.from_dict(data)
    df['opt_coverage'] = opt.opt_val.item()
    for i, player in enumerate(players):
        df['opt_idx_agent{}'.format(i)] = idxfromloc(player.grid_V, opt.opt_goal['Fx_X'][i])
    df['regret'] = df['opt_coverage'] - df['current_coverage']
    file = os.path.join(save_path, 'data.csv')
    df.to_csv(file)


    os.system(
        "cp " + workspace + "/params/" + args.param +
        ".yaml " + save_path + "params.yaml"
    )

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    workspace = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description="A foo that bars")
    parser.add_argument("--param", default="smcc_MacOpt_GP_double")  # params
    parser.add_argument("--env", type=int, default=1)
    parser.add_argument("--i", type=int, default=200)
    args = parser.parse_args()
    train(args)
