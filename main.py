# The environment for this file is a ~/work/rl
import argparse
import errno
import os
import pickle
import warnings

import matplotlib.pyplot as plt
import torch
import yaml

from utils.environement import GridWorld
from utils.ground_truth import GroundTruth
from utils.helper import (SafelyExplore, TrainAndUpdateConstraint,
                          TrainAndUpdateDensity, UpdateCoverageVisu,
                          UpdateSafeVisu, get_frame_writer, idxfromloc,
                          save_data_plots, submodular_optimization)
from utils.initializer import get_players_initialized
from utils.visu import Visu

warnings.filterwarnings("ignore")
plt.rcParams["figure.figsize"] = [12, 6]

workspace = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser(description="A foo that bars")
parser.add_argument("-param", default="smcc_MacOpt_GP")  # params

parser.add_argument("-env", type=int, default=1)
parser.add_argument("-i", type=int, default=200)
args = parser.parse_args()

# 1) Load the config file
with open(workspace + "/params/" + args.param + ".yaml") as file:
    params = yaml.load(file, Loader=yaml.FullLoader)
print(params)

# 2) Set the path and copy params from file
# exp_name = params["experiment"]["name"]
# env_load_path = workspace + "/experiments/" + datetime.today().strftime('%d-%m-%y') + \
#     datetime.today().strftime('-%A')[0:4] + \
#     "/environments/env_" + str(args.env) + "/"
env_load_path = (
    workspace
    + "/experiments/"
    + params["experiment"]["folder"]
    + "/environments/env_"
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

traj_data_dict = {}
for traj_iter in range(params["algo"]["n_CI_samples"]):
    print(args)
    if args.i != -1:
        traj_iter = args.i

    # while running on cluster in parallel sometimes a location in not created if asked by multiple processes
    if not os.path.exists(save_path + str(traj_iter)):
        os.makedirs(save_path + str(traj_iter))

    # lists to record coverage, latter used for regret computations
    list_FxIX_rho_opti = []
    list_FtildexIX_rho_opti = []
    list_FxlIX_lcb_pessi = []
    list_FxlIX_pessi_rho_Rbar0 = []
    list_FxIX_rho_Rbar_eps = []
    list_FxIX_rho_Rbar0 = []
    list_FxIX_lcb_pessi = []
    exploit_record = []

    # Start from some safe initial states
    train = {}
    train["Fx_X"] = init_safe["loc"]
    train["Fx_Y"] = env.get_multi_density_observation(train["Fx_X"])
    players = get_players_initialized(train, params, grid_V)

    # setup visu
    safe_boundary = train["Cx_X"]
    fig, ax = plt.subplots()
    visu = Visu(
        f_handle=fig,
        constraint=params["common"]["constraint"],
        grid_V=grid_V,
        safe_boundary=train["Cx_X"],
        true_constraint_function=opt.true_constraint_function,
        true_objective_func=opt.true_density,
        opt_goal=opt.opt_goal,
        optimal_feasible_boundary=opt.optimal_feasible_boundary,
        agent_param=params["agent"],
        env_params=params["env"],
        common_params=params["common"],
    )

    # visu.plot_optimal_point()
    # Associate safe nodes in the graph and also unsafe

    for it, player in enumerate(players):
        player.update_Fx_gp_with_current_data()
        player.update_graph(init_safe["idx"][it])
        player.save_posterior_normalization_const()
        player.update_current_location(init_safe["loc"][it])

    associate_dict = {}
    associate_dict[0] = []
    for idx in range(params["env"]["n_players"]):
        associate_dict[0].append(idx)

    pessi_associate_dict = {}
    pessi_associate_dict[0] = []
    for idx in range(params["env"]["n_players"]):
        pessi_associate_dict[0].append(idx)

    writer = get_frame_writer()
    iter = -1
    list_sum_max_density_sigma = []
    max_density_sigma = (
        params["env"]["n_players"] * players[0].posterior_normalization_const
    )
    list_sum_max_density_sigma.append(max_density_sigma)
    pt0 = None
    pt1 = None

    # compute coverage based on the initial location
    if params["experiment"]["generate_regret_plot"]:
        FxIX_rho_opti = opt.compute_cover_xIX_rho_opti(
            players, associate_dict)
        FtildexIX_rho_opti = opt.compute_cover_tildexIX_rho_opti(
            players, associate_dict
        )
        FxlIX_pessi_rho_Rbar0, FxlIX_lcb_pessi = opt.compute_cover_xlIX_rho_pessi(
            players, pessi_associate_dict
        )
        FxIX_lcb_pessi = opt.compute_cover_xIX_lcb_pessi(players)
        FxIX_rho_Rbar_eps = opt.compute_cover_xIX_rho_Rbar(players, "eps")
        FxIX_rho_Rbar0 = opt.compute_cover_xIX_rho_Rbar(players, "0")
        list_FxIX_rho_opti.append(FxIX_rho_opti)
        list_FtildexIX_rho_opti.append(FtildexIX_rho_opti)
        list_FxlIX_lcb_pessi.append(FxlIX_lcb_pessi)
        list_FxlIX_pessi_rho_Rbar0.append(FxlIX_pessi_rho_Rbar0)
        list_FxIX_lcb_pessi.append(FxIX_lcb_pessi)
        list_FxIX_rho_Rbar_eps.append(FxIX_rho_Rbar_eps)
        list_FxIX_rho_Rbar0.append(FxIX_rho_Rbar0)
    max_density_sigma = sum([player.get_max_sigma() for player in players]) # sigma is for objective Fx
    list_sum_max_density_sigma.append(max_density_sigma)
    print(iter, max_density_sigma)

    # 4) Solve the submodular problem and get a next point to go xi* in pessimistic safe set
    associate_dict, pessi_associate_dict, acq_density, M_dist = submodular_optimization(
        players, init_safe, params
    )

    # Compute coverage based on the new location
    if params["experiment"]["generate_regret_plot"]:
        FxIX_rho_opti = opt.compute_cover_xIX_rho_opti(
            players, associate_dict)
        FtildexIX_rho_opti = opt.compute_cover_tildexIX_rho_opti(
            players, associate_dict
        )
        FxlIX_pessi_rho_Rbar0, FxlIX_lcb_pessi = opt.compute_cover_xlIX_rho_pessi(
            players, pessi_associate_dict
        )
        FxIX_lcb_pessi = opt.compute_cover_xIX_lcb_pessi(players)
        FxIX_rho_Rbar_eps = opt.compute_cover_xIX_rho_Rbar(players, "eps")
        FxIX_rho_Rbar0 = opt.compute_cover_xIX_rho_Rbar(players, "0")
        list_FxIX_rho_opti.append(FxIX_rho_opti)
        list_FtildexIX_rho_opti.append(FtildexIX_rho_opti)
        list_FxlIX_lcb_pessi.append(FxlIX_lcb_pessi)
        list_FxlIX_pessi_rho_Rbar0.append(FxlIX_pessi_rho_Rbar0)
        list_FxIX_lcb_pessi.append(FxIX_lcb_pessi)
        list_FxIX_rho_Rbar_eps.append(FxIX_rho_Rbar_eps)
        list_FxIX_rho_Rbar0.append(FxIX_rho_Rbar0)
    max_density_sigma = sum([player.max_density_sigma for player in players])
    list_sum_max_density_sigma.append(max_density_sigma)
    print(iter, max_density_sigma)

    # max_density_sigma = sum(
    #     [player.max_density_sigma for player in players])
    # list_sum_max_density_sigma.append(max_density_sigma)
    # print(iter, max_density_sigma)
    bool_SafeUncertainPt = torch.ones(params["env"]["n_players"]) < 0
    goose_step = 0
    TwoStageRun = False
    with writer.saving(fig, save_path + str(traj_iter) + "/video.mp4", dpi=200):
        visu.UpdateIter(iter, -1)
        for agent_key, player in enumerate(players):
            pt1 = UpdateCoverageVisu(
                agent_key, players, visu, env, acq_density, M_dist, writer, fig, pt1
            )
        while max_density_sigma > params["algo"]["eps_density_thresh"] and iter < params["algo"]["n_iter"]:
            '''
            haitong: main while loop.
            '''
            # if max_density_sigma <= params["algo"]["eps_density_thresh"]:
            #     # change gaol to most uncertain point
            #     for agent_key in torch.arange(params["env"]["n_players"])[
            #         bool_SafeUncertainPt
            #     ].tolist():
            #         players[agent_key].update_next_to_go_loc(
            #             players[agent_key].max_constraint_sigma_goal
            #         )
            #         '''
            #         haitong: next_to_go_loc is the planned_measure_loc. here is set to safe expansion.
            #         If no goose, we will never reach here since the while conditions.
            #         '''

            visu.UpdateIter(iter, -1)
            for agent_key in range(params["env"]["n_players"]):
                xi_star = players[agent_key].get_next_to_go_loc()
                players[agent_key].update_current_location(xi_star)

            '''
            haitong: start of iteration for our case.
            '''
            goose_step = 0
            iter += 1
            visu.UpdateIter(iter, -2)
            for agent_key in range(params["env"]["n_players"]):
                # collection of density once goose let us reach
                reached_pt = players[agent_key].get_next_to_go_loc()
                players[agent_key].update_current_location(reached_pt)
                '''
                haitong: this is where update current location in our case.
                '''
                TrainAndUpdateDensity(
                    reached_pt, agent_key, players, params, env)


            for agent_key, player in enumerate(players):
                player.update_Fx_gp_with_current_data()

            (
                associate_dict,
                pessi_associate_dict,
                acq_density,
                M_dist,
            ) = submodular_optimization(players, init_safe, params)
            for agent_key, player in enumerate(players):
                pt1 = UpdateCoverageVisu(
                    agent_key,
                    players,
                    visu,
                    env,
                    acq_density,
                    M_dist,
                    writer,
                    fig,
                    pt1,
                )
            max_density_sigma = sum(
                [player.max_density_sigma for player in players]
            )
            list_sum_max_density_sigma.append(max_density_sigma)
            print(iter, max_density_sigma)

            if params["experiment"]["generate_regret_plot"]:
                FxIX_rho_Rbar_eps = opt.compute_cover_xIX_rho_Rbar(
                    players, "eps")
                FxIX_rho_Rbar0 = opt.compute_cover_xIX_rho_Rbar(
                    players, "0")
                FxIX_rho_opti = opt.compute_cover_xIX_rho_opti(
                    players, associate_dict
                ) # haitong: for no safety considerations, should be same.
                # TODO: change from true asso to opti asso.
                # FtildexIX_rho_opti = opt.compute_opt_Fx_at_t(players)
                FtildexIX_rho_opti = opt.compute_cover_tildexIX_rho_opti(
                    players, associate_dict
                )
                (
                    FxlIX_pessi_rho_Rbar0,
                    FxlIX_lcb_pessi,
                ) = opt.compute_cover_xlIX_rho_pessi(players, pessi_associate_dict)
                FxIX_lcb_pessi = opt.compute_cover_xIX_lcb_pessi(players)
                list_FxIX_rho_opti.append(FxIX_rho_opti)
                list_FtildexIX_rho_opti.append(FtildexIX_rho_opti)
                list_FxlIX_lcb_pessi.append(FxlIX_lcb_pessi)
                list_FxIX_lcb_pessi.append(FxIX_lcb_pessi)
                list_FxlIX_pessi_rho_Rbar0.append(FxlIX_pessi_rho_Rbar0)
                list_FxIX_rho_Rbar_eps.append(FxIX_rho_Rbar_eps)
                list_FxIX_rho_Rbar0.append(FxIX_rho_Rbar0)

        # Plot the final location after you have converged
        for agent_key, player in enumerate(players):
            player.update_current_location(player.planned_disk_center)
            player.update_next_to_go_loc(player.planned_disk_center)
            if params["agent"]["use_goose"]:
                pt0 = UpdateSafeVisu(
                    agent_key,
                    players,
                    visu,
                    env,
                    writer,
                    associate_dict,
                    pessi_associate_dict,
                    fig,
                    pt0,
                )
            pt1 = UpdateCoverageVisu(
                agent_key, players, visu, env, acq_density, M_dist, writer, fig, pt1
            )

    plt.close()  # close the plt so that next video doesn't get affected
    nodes = {}
    nodes["pessi"] = 0
    nodes["opti"] = 0
    nodes["diff"] = 0
    for batch_key in associate_dict:
        nodes["pessi"] += len(set(players[batch_key].pessimistic_graph.nodes))
        nodes["opti"] += len(set(players[batch_key].optimistic_graph.nodes))
        nodes["diff"] += len(
            set(players[batch_key].optimistic_graph.nodes)
            - set(players[batch_key].pessimistic_graph.nodes)
        )
    print("nodes", nodes)
    samples = {}
    samples["constraint"] = players[0].Cx_X_train.shape[0]
    samples["density"] = players[0].Fx_X_train.shape[0]
    print("measurements", samples)
    normalization_factor = {}
    normalization_factor["Rbar0"] = opt.normalization_Rbar0
    normalization_factor["Rbar_eps"] = opt.normalization_Rbar_eps
    if params["experiment"]["generate_regret_plot"]:
        traj_data_dict[traj_iter] = save_data_plots(
            list_FxIX_rho_opti,
            list_FtildexIX_rho_opti,
            list_FxIX_lcb_pessi,
            list_FxlIX_lcb_pessi,
            list_FxlIX_pessi_rho_Rbar0,
            list_sum_max_density_sigma,
            list_FxIX_rho_Rbar_eps,
            list_FxIX_rho_Rbar0,
            opt.opt_val,
            exploit_record,
            nodes,
            samples,
            normalization_factor,
            save_path + str(traj_iter),
        )
        traj_data_dict[traj_iter]["bounds"] = players[0].record
        a_file = open(save_path + "data.pkl", "wb")
        pickle.dump(traj_data_dict, a_file)
        a_file.close()

os.system(
    "cp " + workspace + "/params/" + args.param +
    ".yaml " + save_path + "params.yaml"
)
