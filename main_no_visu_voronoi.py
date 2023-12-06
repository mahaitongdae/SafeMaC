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
import networkx as nx



def train(args):
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
        # 'sum_max_sigma'     :[],
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

    for it, player in enumerate(players):
        player.update_Fx_gp_with_current_data()
        player.update_graph(init_safe["idx"][it])
        player.save_posterior_normalization_const()  # agent.posterior_normalization_const, max of 2beta*sigma.
        player.initialize_location(init_safe["loc"][it])
        data['idx_agent{}'.format(it)].append(idxfromloc(player.grid_V, player.current_location))
        data['idx_measure{}'.format(it)].append(idxfromloc(player.grid_V, player.current_location))
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

    iter = 0
    doubling_target_iter = 0
    pt1 = None

    regret = 0.
    # compute coverage based on the initial location
    current_coverage = opt.compute_current_multiple_coverage(players, associate_dict)
    data.get('current_coverage').append(current_coverage)
    data.get('iter').append(0)
    data.get('instant_regret').append(opt_coverage - current_coverage)
    regret += opt_coverage - current_coverage
    data.get('regret').append(regret)

    # get sigma_0
    initial_maximum_uncertainty = players[0].get_maximum_uncertainty_whole_graph()

    phase = 'exploration' # exploration or coverage
    epoch = 1
    target_planned = False
    coverage_iter = 0

    while iter < args.iter:
        '''
        haitong: main while loop.
        '''

        if phase == 'exploration':

            # 1. exploration stage
            current_locations = []
            measure_locations = []
            for i, player in enumerate(players):
                # target_reached.append(player.update_current_location())
                player.random_walk()
                current_locations.append(player.current_location)
                measure_location = player.get_measurement_pt_max(idxfromloc(player.grid_V, player.current_location))
                measure_locations.append(measure_location)
                data['idx_agent{}'.format(i)].append(idxfromloc(player.grid_V, player.current_location))
                data['idx_measure{}'.format(i)].append(idxfromloc(player.grid_V, measure_location))

            obs = env.get_multi_density_observation(measure_locations)
            players[0].update_Fx_set(torch.stack(measure_locations), torch.cat(obs))
            players[0].update_Fx()
            maximum_uncertainty = players[0].get_maximum_uncertainty_whole_graph()

            if maximum_uncertainty < (params["algo"]["alpha"] ** epoch) * initial_maximum_uncertainty:
                Fx_model = players[0].update_Fx()
                for i in range(1, len(players)):
                    players[i].Fx_model = Fx_model  # sync all Fx model.
                    # haitong: wierd data dim from previous code.
                    # haitong: Centralized algo currently, so we only add to first agent then sync the FX model.
                phase = 'coverage'
                target_planned = False
            # else:
            #     continue

        elif phase == 'coverage':
            print('coverage phase')

            # for _ in range(int(np.ceil(params["agent"]["beta"]) ** epoch)):
            if not target_planned:
                print('planning target')
                node_idxs = [idxfromloc(p.grid_V, p.current_location) for p in players]
                graph = players[0].base_graph
                center_of_masses = []
                for i in range(10): # voronoi iteration, to be tuned
                    voronoi_partitions = nx.voronoi_cells(graph, node_idxs) # , weight=lambda u ,v, d: 1 if v in node_idxs else 0
                    if i == 0:
                        old_voronoi_centers = list(voronoi_partitions.keys())
                    else:
                        old_voronoi_centers = center_of_masses

                    acq_density = players[0].Fx_model.posterior(players[0].grid_V).mvn.mean.detach().numpy()
                    acq_density = acq_density - acq_density.min()

                    for voronoi_center, voronoi_partition in voronoi_partitions.items():
                        subgraph = graph.subgraph(voronoi_partition)
                        center_of_mass = nx.center(subgraph, weight=
                        lambda u ,v, d: acq_density[u] + acq_density[v] if (v == voronoi_center or u == voronoi_center) else 0.01)
                        center_of_masses.append(center_of_mass[0])
                    if sorted(center_of_masses) == sorted(old_voronoi_centers):
                        print('find voronoi partition at iter {}'.format(i + 1))
                        break

                    # OLD VORONOI IMPLEMENTATION
                    # # acq_density = acq_density -
                    # # TODO: might rescale it to handle coverage formulation issue?
                    # partition_pair = np.random.choice(np.linspace(0, 2, 3), size=(3), replace=False).astype(int)
                    # # partition_pair = int(partition_pair)
                    # union_partitions = set(voronoi_partition[node_idxs[partition_pair[0]]]) | set(voronoi_partition[node_idxs[partition_pair[1]]])
                    # subgraph = graph.subgraph(union_partitions)
                    #
                    # # find partition target
                    #
                    # partition_loss = 1e6
                    # for node_a in subgraph.nodes:
                    #     for node_b in subgraph.nodes:
                    #         loss = 0.
                    #         for node in subgraph.nodes:
                    #             loss += acq_density[node] * min(len(nx.shortest_path(subgraph, source=node, target=node_a)),
                    #                                             len(nx.shortest_path(subgraph, source=node, target=node_b)))
                    #         if loss < partition_loss:
                    #             partition_loss = loss
                    #             target_a = node_a
                    #             target_b = node_b

                    # NEW VORONOI IMPLEMENTATION

                for i, player in enumerate(players):
                    player.set_goal(player.grid_V[center_of_masses[i]])

                # path planning.
                path_len = []
                for player in players:
                    # acq_coverage = torch.stack(list(M_dist[0].values())).detach().numpy()
                    # acq_coverage = acq_coverage - acq_coverage.min() # to handle negative edge weights in planning.
                    player.shortest_path_planning() # directly use shortest path planning here.
                    path_len.append(len(player.path))
                print('max path len {}'.format(max(path_len)))
                target_planned = True
                # compute target doubling trick iter.
                # planned_target_samples = []
                # for i, player in enumerate(players):
                #     planned_target_samples.append(
                #         data['idx_agent{}'.format(i)].count(idxfromloc(player.grid_V, player.planned_disk_center)))
                # min_samples = min(planned_target_samples)
                # doubling_target_iter = iter + min_samples
                # print('double until iter {}'.format(doubling_target_iter))

            if target_planned:
                    target_reached = []
                    measure_locations = []
                    for i, player in enumerate(players):
                        target_reached.append(player.update_current_location())
                        measure_location = player.get_measurement_pt_max(
                            idxfromloc(player.grid_V, player.current_location))
                        measure_locations.append(measure_location)
                        data['idx_agent{}'.format(i)].append(idxfromloc(player.grid_V, player.current_location))
                        data['idx_measure{}'.format(i)].append(idxfromloc(player.grid_V, measure_location))

                    obs = env.get_multi_density_observation(measure_locations)
                    players[0].update_Fx_set(torch.stack(measure_locations), torch.cat(obs))

                    if all(target_reached):
                        Fx_model = players[0].update_Fx()
                        for i in range(1, len(players)):
                            players[i].Fx_model = Fx_model  # sync all Fx model.
                        # phase = 'exploration'
                        target_planned = False
                    coverage_iter += 1
            if coverage_iter >= (int(np.ceil(params["algo"]["beta"]) ** epoch)):
                coverage_iter = 0
                phase = 'exploration'
                epoch += 1
                print(f"start epoch {epoch}")


        iter += 1

        current_coverage = opt.compute_current_multiple_coverage(players, associate_dict)
        data.get('current_coverage').append(current_coverage)
        data.get('instant_regret').append(opt_coverage - current_coverage)
        regret += opt_coverage - current_coverage
        data.get('regret').append(regret)
        data.get('iter').append(iter)
        print("Iter: {}, coverage value: {:.3f}".format(iter, current_coverage))

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

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    workspace = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description="A foo that bars")
    parser.add_argument("--param", default="GP_base_voronoi")  # params
    parser.add_argument("--env_idx", type=int, default=100)
    parser.add_argument("--generate", type=bool, default=False)
    parser.add_argument("--noise_sigma", type=float, default=0.01)
    parser.add_argument("--iter", type=int, default=100)
    args = parser.parse_args()
    train(args)
