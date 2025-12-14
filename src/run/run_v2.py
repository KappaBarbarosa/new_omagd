import datetime
import os
import pprint
import time
import threading
import torch as th
from types import SimpleNamespace as SN
from utils.logging import Logger
from utils.timehelper import time_left, time_str
from os.path import dirname, abspath
import sys

from learners import REGISTRY as le_REGISTRY
from runners import REGISTRY as r_REGISTRY
from controllers import REGISTRY as mac_REGISTRY
from components.episode_buffer import ReplayBuffer
from components.transforms import OneHot

from tqdm import tqdm


def run(_run, _config, _log):
    # check args sanity
    _config = args_sanity_check(_config, _log)

    args = SN(**_config)

    th.set_num_threads(args.thread_num)
    # th.set_num_interop_threads(8)

    args.device = "cuda" if args.use_cuda else "cpu"

    # setup loggers
    logger = Logger(_log)

    _log.info("Experiment Parameters:")
    experiment_params = pprint.pformat(_config,
                                       indent=4,
                                       width=1)
    _log.info("\n\n" + experiment_params + "\n")

    # configure tensorboard logger
    unique_token = "{}__{}".format(args.name, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    args.unique_token = unique_token

    testing_algorithms = ["vdn", "qmix", "hpn_vdn", "hpn_qmix",
                          "deepset_vdn", "deepset_qmix", "deepset_hyper_vdn", "deepset_hyper_qmix",
                          "updet_vdn", "updet_qmix", "vdn_DA", "qmix_DA",
                          "gnn_vdn", "gnn_qmix", "qplex", "hpn_qplex", "asn"
                          ]
    env_name = args.env
    logdir = env_name
    if env_name in ["sc2", "sc2_v2", ]:
        logdir = os.path.join("{}_{}-obs_aid={}-obs_act={}".format(
            logdir,
            args.env_args["map_name"],
            int(args.obs_agent_id),
            int(args.obs_last_action),
        ))
        if env_name == "sc2_v2":
            logdir = logdir + "-conic_fov={}".format(
                "1-change_fov_by_move={}".format(
                    int(args.env_args["change_fov_with_move"])) if args.env_args["conic_fov"] else "0"
            )
    logdir = os.path.join(logdir,
                          "algo={}-agent={}".format(args.name, args.agent),
                          "env_n={}".format(
                              args.batch_size_run,
                          ))
    if args.name in testing_algorithms:
        if args.name in ["vdn_DA", "qmix_DA", ]:
            logdir = os.path.join(logdir,
                                  "{}-data_augment={}".format(
                                      args.mixer, args.augment_times
                                  ))
        elif args.name in ["gnn_vdn", "gnn_qmix"]:
            logdir = os.path.join(logdir,
                                  "{}-layer_num={}".format(
                                      args.mixer, args.gnn_layer_num
                                  ))
        elif args.name in ["vdn", "qmix", "deepset_vdn", "deepset_qmix", "qplex", "asn"]:
            logdir = os.path.join(logdir,
                                  "mixer={}".format(
                                      args.mixer,
                                  ))
        elif args.name in ["updet_vdn", "updet_qmix"]:
            logdir = os.path.join(logdir,
                                  "mixer={}-att_dim={}-att_head={}-att_layer={}".format(
                                      args.mixer,
                                      args.transformer_embed_dim,
                                      args.transformer_heads,
                                      args.transformer_depth,
                                  ))
        elif args.name in ["deepset_hyper_vdn", "deepset_hyper_qmix"]:
            logdir = os.path.join(logdir,
                                  "mixer={}-hpn_hyperdim={}".format(
                                      args.mixer,
                                      args.hpn_hyper_dim,
                                  ))
        elif args.name in ["hpn_vdn", "hpn_qmix", "hpn_qplex"]:
            logdir = os.path.join(logdir,
                                  "head_n={}-mixer={}-hpn_hyperdim={}-acti={}".format(
                                      args.hpn_head_num,
                                      args.mixer,
                                      args.hpn_hyper_dim,
                                      args.hpn_hyper_activation,
                                  ))

    logdir = os.path.join(logdir,
                          "rnn_dim={}-2bs={}_{}-tdlambda={}-epdec_{}={}k".format(
                              args.rnn_hidden_dim,
                              args.buffer_size,
                              args.batch_size,
                              args.td_lambda,
                              args.epsilon_finish,
                              args.epsilon_anneal_time // 1000,
                          ))
    args.log_model_dir = logdir
    if args.use_tensorboard:
        tb_logs_direc = os.path.join(dirname(dirname(dirname(abspath(__file__)))), args.local_results_path, "tb_logs")
        tb_exp_direc = os.path.join(tb_logs_direc, "{}").format(unique_token)
        if args.name in testing_algorithms:  # add parameter config to the logger path！
            tb_exp_direc = os.path.join(tb_logs_direc, logdir, unique_token)
        logger.setup_tb(tb_exp_direc)

    # sacred is on by default
    logger.setup_sacred(_run)

    # configure wandb logger
    if getattr(args, 'use_wandb', False):
        wandb_config = _config.copy()
        
        # Use different project name for reconstruction pretraining
        if getattr(args, 'use_graph_reconstruction', False):
            map_name = args.env_args.get('map_name', 'unknown')
            stage = getattr(args, 'recontructer_stage', 'stage1')
            project_name = f"{map_name}_reconstruction_{stage}"
        else:
            project_name = f"{map_name}_exp"
        
        wandb_kwargs = {
            'project': project_name,
            'name': unique_token,
            'config': wandb_config,
        }
        if hasattr(args, 'wandb_entity') and args.wandb_entity:
            wandb_kwargs['entity'] = args.wandb_entity
        logger.setup_wandb(**wandb_kwargs)

    # Run and train
    run_sequential(args=args, logger=logger)

    # Clean up after finishing
    print("Exiting Main")

    print("Stopping all threads")
    for t in threading.enumerate():
        if t.name != "MainThread":
            print("Thread {} is alive! Is daemon: {}".format(t.name, t.daemon))
            t.join(timeout=1)
            print("Thread joined")

    print("Exiting script")

    # Making sure framework really exits
    os._exit(os.EX_OK)


def evaluate_sequential(args, runner):
    for _ in range(args.test_nepisode):
        runner.run(test_mode=True)

    if args.save_replay:
        runner.save_replay()

    runner.close_env()


def pretrain_graph_reconstructer(args, runner, learner, buffer, logger):
    """
    Pretrain graph reconstructer before MAC training.
    
    Stage 1: Train tokenizer (VQ-VAE)
    Stage 2: Train mask predictor (with frozen tokenizer)
    """
    stage = getattr(args, 'recontructer_stage', 'stage1')
    pretrain_episodes = getattr(args, 'graph_pretrain_episodes', args.buffer_size)
    pretrain_epochs = getattr(args, 'graph_pretrain_epochs', 100)
    pretrain_batch_size = getattr(args, 'graph_pretrain_batch_size', args.batch_size)
    log_interval = getattr(args, 'graph_pretrain_log_interval', 10)
    eval_interval = getattr(args, 'graph_pretrain_eval_interval', 20)
    eval_episodes = getattr(args, 'graph_pretrain_eval_episodes', args.batch_size_run * 5)
    
    logger.console_logger.info(\"=\" * 60)
    logger.console_logger.info(f"=== Graph Reconstructer Pretrain: {stage} ===")
    logger.console_logger.info(f"    Episodes to collect: {pretrain_episodes}")
    logger.console_logger.info(f"    Training epochs: {pretrain_epochs}")
    logger.console_logger.info(f"    Batch size: {pretrain_batch_size}")
    logger.console_logger.info(f"    Eval interval: {eval_interval} epochs")
    logger.console_logger.info(f"    Eval episodes: {eval_episodes}")
    logger.console_logger.info("=" * 60)
    
    # Phase 1: Collect training episodes
    logger.console_logger.info(f"[Pretrain] Collecting {pretrain_episodes} training episodes...")
    collected = 0

    with tqdm(total=pretrain_episodes, initial=buffer.episodes_in_buffer, desc="Collecting pretrain episodes") as pbar:
        while buffer.episodes_in_buffer < pretrain_episodes:
            with th.no_grad():
                episode_batch = runner.run(test_mode=False, skip_logging=True)
                if episode_batch.batch_size > 0:
                    buffer.insert_episode_batch(episode_batch)
                    pbar.update(episode_batch.batch_size)
    logger.console_logger.info(f"[Pretrain] Training data collection complete: {buffer.episodes_in_buffer} episodes")
    
    # Track best eval loss for model saving
    best_eval_loss = float('inf')
    best_model_path = None
    
    # Phase 2: Train graph reconstructer with periodic evaluation
    logger.console_logger.info(f"[Pretrain] Training {stage} for {pretrain_epochs} epochs...")
    for epoch in range(pretrain_epochs):
        if not buffer.can_sample(pretrain_batch_size):
            logger.console_logger.warning("[Pretrain] Not enough data to sample, skipping epoch")
            continue
            
        episode_sample = buffer.sample(pretrain_batch_size)
        max_ep_t = episode_sample.max_t_filled()
        episode_sample = episode_sample[:, :max_ep_t]
        
        if episode_sample.device != args.device:
            episode_sample.to(args.device)
        
        # Train graph reconstructer
        learner.train_graph_reconstructor(episode_sample, runner.t_env, epoch)
        
        # Evaluation: collect NEW episodes and compute eval loss
        if (epoch + 1) % eval_interval == 0 or epoch == pretrain_epochs - 1:
            eval_loss, eval_metrics = evaluate_graph_reconstructer(
                args, runner, learner, logger, eval_episodes, epoch
            )
            
            # Save best model
            if eval_loss < best_eval_loss and args.save_model:
                best_eval_loss = eval_loss
                best_model_path = os.path.join(
                    args.local_results_path, "models", args.log_model_dir,
                    args.unique_token, f"pretrain_{stage}_best"
                )
                os.makedirs(best_model_path, exist_ok=True)
                learner.save_graph_reconstructor(best_model_path)
                logger.console_logger.info(f"[Pretrain] New best model saved (loss: {eval_loss:.4f})")
    
    logger.console_logger.info(f"[Pretrain] {stage} training completed!")
    
    # Save final model
    if args.save_model:
        save_path = os.path.join(
            args.local_results_path, "models", args.log_model_dir, 
            args.unique_token, f"pretrain_{stage}"
        )
        os.makedirs(save_path, exist_ok=True)
        learner.save_graph_reconstructor(save_path)
        logger.console_logger.info(f"[Pretrain] Saved final {stage} model to {save_path}")
        if best_model_path:
            logger.console_logger.info(f"[Pretrain] Best model saved at {best_model_path} (loss: {best_eval_loss:.4f})")
    
    logger.console_logger.info("=" * 60)


def evaluate_graph_reconstructer(args, runner, learner, logger, eval_episodes, epoch):
    """
    Evaluate graph reconstructer using newly collected episodes.
    
    Returns:
        eval_loss: Average evaluation loss
        eval_metrics: Dictionary of evaluation metrics
    """
    logger.console_logger.info(f"[Eval] Collecting {eval_episodes} new episodes for evaluation...")
    
    # Collect fresh evaluation data
    eval_batches = []
    collected = 0
    while collected < eval_episodes:
        with th.no_grad():
            episode_batch = runner.run(test_mode=True, skip_logging=True)  # Use test_mode for eval
            if episode_batch.batch_size > 0:
                eval_batches.append(episode_batch)
                collected += episode_batch.batch_size
    
    # Compute evaluation loss on fresh data
    total_loss = 0.0
    all_metrics = {}
    num_batches = len(eval_batches)
    
    with th.no_grad():
        for batch in eval_batches:
            max_ep_t = batch.max_t_filled()
            batch = batch[:, :max_ep_t]
            
            if batch.device != args.device:
                batch.to(args.device)
            
            loss, metrics = learner.eval_graph_reconstructor(batch)
            total_loss += loss
            
            # Accumulate metrics
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    if key not in all_metrics:
                        all_metrics[key] = 0.0
                    all_metrics[key] += value
    
    # Average metrics
    avg_loss = total_loss / max(num_batches, 1)
    avg_metrics = {k: v / max(num_batches, 1) for k, v in all_metrics.items()}
    
    # Log evaluation results - use epoch as step for better W&B visualization
    logger.console_logger.info(f"[Eval] Epoch {epoch} - Eval Loss: {avg_loss:.4f}")
    logger.log_stat("eval/graph_loss", avg_loss, epoch)
    
    # Log additional metrics to both file and console
    for key, value in avg_metrics.items():
        if isinstance(value, (int, float)):
            logger.log_stat(f"eval/_{key}", value, epoch)
    
    # Print key evaluation metrics to console 
    if avg_metrics:
        metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in avg_metrics.items() 
                                  if k in ['mse', 'cosine_similarity', 'feature_correlation', 'perplexity', 'codebook_usage']])
        if metrics_str:
            logger.console_logger.info(f"[Eval]   Metrics: {metrics_str}")
    
    # Display sample comparisons from last batch
    if eval_batches and 'samples' in metrics:
        from modules.graph_reconstructers.tokenizer_logger import format_sample_comparison
        sample_str = format_sample_comparison(metrics.get('samples', []))
        if sample_str:
            logger.console_logger.info(f"[Eval]   Sample Reconstructions:\n{sample_str}")
    
    return avg_loss, avg_metrics


def run_sequential(args, logger):
    # Init runner so we can get env info
    runner = r_REGISTRY[args.runner](args=args, logger=logger)

    # Set up schemes and groups here
    env_info = runner.get_env_info()
    args.n_agents = env_info["n_agents"]
    args.n_actions = env_info["n_actions"]
    args.state_shape = env_info["state_shape"]
    args.obs_shape = env_info["obs_shape"]
    args.accumulated_episodes = getattr(args, "accumulated_episodes", None)

    if args.env in ["sc2", "sc2_v2", "gfootball"]:
        if args.env in ["sc2", "sc2_v2"]:
            args.output_normal_actions = env_info["n_normal_actions"]
        args.n_enemies = env_info["n_enemies"]
        args.n_allies = env_info["n_allies"]
        # args.obs_ally_feats_size = env_info["obs_ally_feats_size"]
        # args.obs_enemy_feats_size = env_info["obs_enemy_feats_size"]
        args.state_ally_feats_size = env_info["state_ally_feats_size"]
        args.state_enemy_feats_size = env_info["state_enemy_feats_size"]
        args.obs_component = env_info["obs_component"]
        args.state_component = env_info["state_component"]
        args.map_type = env_info["map_type"]
        args.agent_own_state_size = env_info["state_ally_feats_size"]

    # Default/Base scheme
    scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        "full_obs": {"vshape": env_info["obs_shape"], "group": "agents"},  # Full observation (no range limit)
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.int},
        "probs": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.float},
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
    }
    groups = {
        "agents": args.n_agents
    }
    preprocess = {
        "actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])
    }
    # [batch, episode_length, n_agents, feature_dim]
    buffer = ReplayBuffer(scheme, groups, args.buffer_size, env_info["episode_limit"] + 1,
                          preprocess=preprocess,
                          device="cpu" if args.buffer_cpu_only else args.device)
    # Setup multiagent controller here
    mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args)

    # Give runner the scheme
    runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)

    # Learner
    learner = le_REGISTRY[args.learner](mac, buffer.scheme, logger, args)

    if args.use_cuda:
        learner.cuda()

    if args.checkpoint_path != "":
        timesteps = []
        timestep_to_load = 0

        if not os.path.isdir(args.checkpoint_path):
            logger.console_logger.info("Checkpoint directiory {} doesn't exist".format(args.checkpoint_path))
            return

        # Go through all files in args.checkpoint_path
        for name in os.listdir(args.checkpoint_path):
            full_name = os.path.join(args.checkpoint_path, name)
            # Check if they are dirs the names of which are numbers
            if os.path.isdir(full_name) and name.isdigit():
                timesteps.append(int(name))

        if args.load_step == 0:
            # choose the max timestep
            timestep_to_load = max(timesteps)
        else:
            # choose the timestep closest to load_step
            timestep_to_load = min(timesteps, key=lambda x: abs(x - args.load_step))

        model_path = os.path.join(args.checkpoint_path, str(timestep_to_load))

        logger.console_logger.info("Loading model from {}".format(model_path))
        learner.load_models(model_path)
        runner.t_env = timestep_to_load

        if args.evaluate or args.save_replay:
            evaluate_sequential(args, runner)
            return

    # start training
    episode = 0
    last_test_T = -args.test_interval - 1
    last_log_T = 0
    model_save_time = 0

    start_time = time.time()
    last_time = start_time

    # ========== Graph Reconstructer Pretrain ==========
    if getattr(args, 'use_graph_reconstruction', False):
        pretrain_graph_reconstructer(args, runner, learner, buffer, logger)

    # logger.console_logger.info("Beginning training for {} timesteps".format(args.t_max))

    # while runner.t_env <= args.t_max:
    #     # Run for a whole episode at a time
    #     with th.no_grad():
    #         # t_start = time.time()
    #         episode_batch = runner.run(test_mode=False)
    #         if episode_batch.batch_size > 0:  # After clearing the batch data, the batch may be empty.
    #             buffer.insert_episode_batch(episode_batch)
    #         # print("Sample new batch cost {} seconds.".format(time.time() - t_start))
    #         episode += args.batch_size_run

    #     if buffer.can_sample(args.batch_size):
    #         if args.accumulated_episodes and episode % args.accumulated_episodes != 0:
    #             continue

    #         episode_sample = buffer.sample(args.batch_size)

    #         # Truncate batch to only filled timesteps
    #         max_ep_t = episode_sample.max_t_filled()
    #         episode_sample = episode_sample[:, :max_ep_t]

    #         if episode_sample.device != args.device:
    #             episode_sample.to(args.device)

    #         learner.train(episode_sample, runner.t_env, episode)
    #         del episode_sample

    #     # Execute test runs once in a while
    #     n_test_runs = max(1, args.test_nepisode // runner.batch_size)
    #     if (runner.t_env - last_test_T) / args.test_interval >= 1.0:
    #         logger.console_logger.info("t_env: {} / {}".format(runner.t_env, args.t_max))
    #         logger.console_logger.info("Estimated time left: {}. Time passed: {}".format(
    #             time_left(last_time, last_test_T, runner.t_env, args.t_max), time_str(time.time() - start_time)))
    #         last_time = time.time()
    #         last_test_T = runner.t_env
    #         with th.no_grad():
    #             for _ in range(n_test_runs):
    #                 runner.run(test_mode=True)

    #     if args.save_model and (
    #             runner.t_env - model_save_time >= args.save_model_interval or runner.t_env >= args.t_max):
    #         model_save_time = runner.t_env
    #         save_path = os.path.join(args.local_results_path, "models", args.log_model_dir, args.unique_token,
    #                                  str(runner.t_env))
    #         # "results/models/{}".format(unique_token)
    #         os.makedirs(save_path, exist_ok=True)
    #         logger.console_logger.info("Saving models to {}".format(save_path))

    #         # learner should handle saving/loading -- delegate actor save/load to mac,
    #         # use appropriate filenames to do critics, optimizer states
    #         learner.save_models(save_path)

    #     if (runner.t_env - last_log_T) >= args.log_interval:
    #         logger.log_stat("episode", episode, runner.t_env)
    #         logger.log_stat("episode_in_buffer", buffer.episodes_in_buffer, runner.t_env)
    #         logger.print_recent_stats()
    #         last_log_T = runner.t_env

    runner.close_env()
    logger.console_logger.info("Finished Training")

    # flush
    sys.stdout.flush()
    time.sleep(10)


def args_sanity_check(config, _log):
    # set CUDA flags
    # config["use_cuda"] = True # Use cuda whenever possible!
    if config["use_cuda"] and not th.cuda.is_available():
        config["use_cuda"] = False
        _log.warning("CUDA flag use_cuda was switched OFF automatically because no CUDA devices are available!")

    if config["test_nepisode"] < config["batch_size_run"]:
        config["test_nepisode"] = config["batch_size_run"]
    else:
        config["test_nepisode"] = (config["test_nepisode"] // config["batch_size_run"]) * config["batch_size_run"]

    return config
