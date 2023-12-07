# The environment for this file is a ~/work/rl
import argparse
import copy
import errno
import os
import pickle
import warnings
from multiprocessing import Process, Lock, Array, Manager
import time

import matplotlib.pyplot as plt
import torch
import yaml
import numpy as np
import pandas as pd

import sys
sys.path.append(os.path.dirname(__file__))
import networkx as nx

from utils.environement import GridWorld
from utils.ground_truth import GroundTruth
from utils.helper import submodular_optimization, idxfromloc
from utils.initializer import get_players_initialized
from utils.visualizer_simu import Visulizer

def train(args, shared_locs, lock):
    time.sleep(3) # sleep to wait visulizer
    env, agent, algo = args.param.split('_')
    params = {}
    env = 'env_' + env
    agent = 'agent_' + agent
    algo = 'algo_' + algo
    # 1) Load the config file
    for param in [env, agent, algo]:
        with open(workspace + "/params/" + param + ".yaml") as file:
            params.update(yaml.load(file, Loader=yaml.FullLoader))
    print(params)

    if not args.generate:
        params["env"].update({"generate": False})

    params["env"].update({"Fx_noise" : args.noise_sigma})
    params["agent"].update({"Fx_noise": args.noise_sigma})

    env_load_path = (
        workspace
        + "/experiments/"
        + params["experiment"]["folder"] + '_' + str(args.noise_sigma)
        + "/"
        + env + '_'
        + str(args.env_idx)
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
    opt_coverage = opt.opt_val.item()

    # while running on cluster in parallel sometimes a location in not created if asked by multiple processes
    os.makedirs(save_path, exist_ok=True)

    # use df to store data.
    data = {
        'current_coverage': [],
        'iter': [],
        'instant_regret': [],
        'regret': []
    }
    data.update({'idx_agent{}'.format(i): [] for i in range(params["env"]["n_players"])})
    data.update({'idx_measure{}'.format(i): [] for i in range(params["env"]["n_players"])})




    # Start from some safe initial states
    train = {}
    train["Cx_X"] = init_safe["loc"]
    train["Cx_Y"] = env.get_multi_constraint_observation(train["Cx_X"])
    train["Fx_X"] = init_safe["loc"]
    train["Fx_Y"] = env.get_multi_density_observation(train["Fx_X"])
    players = get_players_initialized(train, params, grid_V)
    env_graph = copy.deepcopy(players[0].base_graph)
    nx.set_node_attributes(env_graph, 0, 'visit_count')

    # get initial visit count
    visit_count_last_ep = nx.get_node_attributes(env_graph, 'visit_count')

    for it, player in enumerate(players):
        player.update_Fx_gp_with_current_data()
        player.update_graph(init_safe["idx"][it])
        player.save_posterior_normalization_const()  # agent.posterior_normalization_const, max of 2beta*sigma.
        player.initialize_location(init_safe["loc"][it])
        current_loc_idx = idxfromloc(player.grid_V, player.current_location)
        data['idx_agent{}'.format(it)].append(current_loc_idx)
        data['idx_measure{}'.format(it)].append(current_loc_idx)
        env_graph.nodes[current_loc_idx]["visit_count"] += 1
        # haitong: will not do measurement on the initialization. Use current for in-place.
        # measure_loc = player.get_measurement_pt_max(idxfromloc(player.grid_V, player.current_location))


    associate_dict = {}
    associate_dict[0] = []
    for idx in range(params["env"]["n_players"]):
        associate_dict[0].append(idx)

    pessi_associate_dict = {}
    pessi_associate_dict[0] = []
    for idx in range(params["env"]["n_players"]):
        pessi_associate_dict[0].append(idx)

    main_iter = 0
    regret = 0.
    # compute coverage based on the initial location
    current_coverage = opt.compute_current_multiple_coverage(players, associate_dict)
    data.get('current_coverage').append(current_coverage)
    data.get('iter').append(0)
    data.get('instant_regret').append(opt_coverage - current_coverage)
    regret += opt_coverage - current_coverage
    data.get('regret').append(regret)


    # 4) Solve the submodular problem and get a next point to go xi* in pessimistic safe set
    associate_dict, pessi_associate_dict, acq_density, M_dist = submodular_optimization(
        players, init_safe, params
    )
    '''
    haitong: in submodular_optimization, goal is set by agent.planned_dist_center & agent.planned_measured_loc.
                we keep both for compatible with previous visualization code.
    '''
    acq_coverage = torch.stack(list(M_dist[0].values())).detach().numpy()
    for player in players:
        # player.planning(acq_coverage)
        player.shortest_path_planning()

    while main_iter < args.iter:
        '''
        haitong: main while loop.
        '''
        time.sleep(1)

        target_reached = []
        current_locations = []
        measure_locations = []
        for i, player in enumerate(players):
            target_reached.append(player.update_current_location())

            # update 
            with lock:
                shared_locs[i] = idxfromloc(player.grid_V, player.current_location)

            current_locations.append(player.current_location)
            measure_location = player.get_measurement_pt_max(idxfromloc(player.grid_V, player.current_location))
            measure_locations.append(measure_location)
            current_loc_idx = idxfromloc(player.grid_V, player.current_location)
            data['idx_agent{}'.format(i)].append(current_loc_idx)
            data['idx_measure{}'.format(i)].append(idxfromloc(player.grid_V, measure_location))
            env_graph.nodes[current_loc_idx]['visit_count'] += 1

        obs = env.get_multi_density_observation(measure_locations)
        players[0].update_Fx_set(torch.stack(measure_locations), torch.cat(obs))
        # haitong: wierd data dim from previous code.
        # haitong: Centralized algo currently, so we only add to first agent then sync the FX model.

        replan = False
        # doubling trick condition:
        if params["algo"]["use_doubling_trick"]:
            visit_count_current = nx.get_node_attributes(env_graph, 'visit_count')
            for key in visit_count_current.keys():
                if visit_count_current[key] >= max(2*visit_count_last_ep[key], 1):
                    print('replan index {}, previous ep {}, current {}'.format(key, visit_count_last_ep[key], visit_count_current[key]))
                    replan = True
                    break
        elif not params["algo"]["use_doubling_trick"]:
            replan = all(target_reached) # for baseline, replan until target reached.
        else:
            raise NotImplementedError


        if replan:
            # if finish doubling trick this time, update GP_0.01 and do planning.
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
                acq_coverage = acq_coverage - acq_coverage.min() # to handle negative edge weights in planning.
                player.planning(acq_coverage)
                path_len.append(len(player.path))
            print('max path len {}'.format(max(path_len)))

            visit_count_last_ep = nx.get_node_attributes(env_graph, 'visit_count')

        main_iter += 1

        current_coverage = opt.compute_current_multiple_coverage(players, associate_dict)
        data.get('current_coverage').append(current_coverage)
        data.get('instant_regret').append(opt_coverage - current_coverage)
        regret += opt_coverage - current_coverage
        data.get('regret').append(regret)
        data.get('iter').append(main_iter)
        print("Iter: {}, coverage value: {:.3f}".format(main_iter, current_coverage))

    df = pd.DataFrame.from_dict(data)
    df['opt_coverage'] = opt_coverage
    for i, player in enumerate(players):
        df['opt_idx_agent{}'.format(i)] = idxfromloc(player.grid_V, opt.opt_goal['Fx_X'][i])
    # df['regret'] = df['opt_coverage'] - df['current_coverage']
    file = os.path.join(save_path, 'data.csv')
    df.to_csv(file)


    # os.system(
    #     "cp " + workspace + "/params/" + args.param +
    #     ".yaml " + save_path + "params.yaml"
    # )

    save_file = os.path.join(save_path, 'params.yaml')
    with open(save_file, 'w') as f:
        yaml.dump(params, f)

def run_visualizer(shared_list, lock):
    visu = Visulizer(shared_list, lock)
    visu.run()

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    workspace = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description="A foo that bars")
    parser.add_argument("--param", default="real_base_double")  # params
    parser.add_argument("--env_idx", type=int, default=100)
    parser.add_argument("--generate", action='store_true')
    parser.add_argument("--noise_sigma", type=float, default=0.01)
    parser.add_argument("--iter", type=int, default=200)
    parser.set_defaults(generate=False)
    args = parser.parse_args()
    
    shared_list = Manager().list([0, 1, 2])
    lock = Lock()
    pt = Process(target=train, args=(args, shared_list, lock))
    pv = Process(target=run_visualizer, args=(shared_list, lock))
    pt.start()
    pv.start()
    pt.join()
    pv.join()

    # train(args)
