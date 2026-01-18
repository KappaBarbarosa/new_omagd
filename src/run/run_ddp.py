"""
DDP-enabled run script for multi-GPU distributed training.
Supports both Stage 1/2 pretraining and Stage 3 QMIX training.
"""
import datetime
import os
import pprint
import time
import threading
import torch as th
import torch.distributed as dist
import torch.multiprocessing as mp
from types import SimpleNamespace as SN
from utils.logging import Logger
from utils.timehelper import time_left, time_str
from os.path import dirname, abspath
import sys
from collections import Counter

from learners import REGISTRY as le_REGISTRY
from runners import REGISTRY as r_REGISTRY
from controllers import REGISTRY as mac_REGISTRY
from components.episode_buffer import ReplayBuffer
from components.transforms import OneHot

from utils.dist_utils import (
    setup_distributed, cleanup_distributed, is_main_process, 
    get_rank, get_world_size
)

from tqdm import tqdm


def run(_run, _config, _log):
    """
    Main entry point for DDP training.
    This spawns multiple processes for multi-GPU training.
    """
    _config = args_sanity_check(_config, _log)
    
    # Check if DDP is enabled
    use_ddp = _config.get('use_ddp', False)
    world_size = _config.get('world_size', th.cuda.device_count())
    
    if use_ddp and world_size > 1:
        # Spawn processes for DDP training
        _log.info(f"Starting DDP training with {world_size} GPUs")
        mp.spawn(
            ddp_worker,
            args=(_run, _config, _log, world_size),
            nprocs=world_size,
            join=True
        )
    else:
        # Single GPU or CPU training
        _log.info("Running single-GPU training (DDP disabled or world_size=1)")
        run_single(_run, _config, _log)


def ddp_worker(rank, _run, _config, _log, world_size):
    """
    Worker function for each DDP process.
    
    Args:
        rank: Process rank (GPU index)
        _run: Sacred run object
        _config: Configuration dictionary
        _log: Logger
        world_size: Total number of GPUs
    """
    # Setup distributed environment
    port = _config.get('ddp_port', 12355)
    setup_distributed(rank, world_size, backend='nccl', port=port)
    
    try:
        # Run training on this GPU
        run_single(_run, _config, _log, rank=rank)
    finally:
        cleanup_distributed()


def run_single(_run, _config, _log, rank=0):
    """
    Run training on a single GPU (called by each DDP process).
    
    Args:
        _run: Sacred run object
        _config: Configuration dictionary
        _log: Logger
        rank: GPU rank (0 for single-GPU training)
    """
    args = SN(**_config)
    
    # Set device
    if args.use_cuda:
        th.cuda.set_device(rank)
        args.device = f"cuda:{rank}"
    else:
        args.device = "cpu"
    
    th.set_num_threads(args.thread_num)
    
    is_main = is_main_process()
    
    # Only main process sets up logging
    if is_main:
        logger = Logger(_log)
        _log.info("Experiment Parameters:")
        experiment_params = pprint.pformat(_config, indent=4, width=1)
        _log.info("\n\n" + experiment_params + "\n")
    else:
        # Create a minimal logger for non-main processes
        logger = Logger(_log, minimal=True)
    
    # Configure unique token and directories
    unique_token = "{}__{}".format(args.name, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    args.unique_token = unique_token
    
    # Build log directory
    testing_algorithms = ["vdn", "qmix", "hpn_vdn", "hpn_qmix",
                          "deepset_vdn", "deepset_qmix", "deepset_hyper_vdn", "deepset_hyper_qmix",
                          "updet_vdn", "updet_qmix", "vdn_DA", "qmix_DA",
                          "gnn_vdn", "gnn_qmix", "qplex", "hpn_qplex", "asn"]
    env_name = args.env
    logdir = env_name
    if env_name in ["sc2", "sc2_v2"]:
        logdir = os.path.join("{}_{}-obs_aid={}-obs_act={}".format(
            logdir, args.env_args["map_name"],
            int(args.obs_agent_id), int(args.obs_last_action),
        ))
        if env_name == "sc2_v2":
            logdir = logdir + "-conic_fov={}".format(
                "1-change_fov_by_move={}".format(
                    int(args.env_args["change_fov_with_move"])) if args.env_args["conic_fov"] else "0"
            )
    logdir = os.path.join(logdir,
                          "algo={}-agent={}".format(args.name, args.agent),
                          "env_n={}".format(args.batch_size_run))
    if args.name in testing_algorithms:
        if args.name in ["vdn_DA", "qmix_DA"]:
            logdir = os.path.join(logdir, "{}-data_augment={}".format(args.mixer, args.augment_times))
        elif args.name in ["gnn_vdn", "gnn_qmix"]:
            logdir = os.path.join(logdir, "{}-layer_num={}".format(args.mixer, args.gnn_layer_num))
        elif args.name in ["vdn", "qmix", "deepset_vdn", "deepset_qmix", "qplex", "asn"]:
            logdir = os.path.join(logdir, "mixer={}".format(args.mixer))
        elif args.name in ["updet_vdn", "updet_qmix"]:
            logdir = os.path.join(logdir, "mixer={}-att_dim={}-att_head={}-att_layer={}".format(
                args.mixer, args.transformer_embed_dim, args.transformer_heads, args.transformer_depth))
        elif args.name in ["deepset_hyper_vdn", "deepset_hyper_qmix"]:
            logdir = os.path.join(logdir, "mixer={}-hpn_hyperdim={}".format(args.mixer, args.hpn_hyper_dim))
        elif args.name in ["hpn_vdn", "hpn_qmix", "hpn_qplex"]:
            logdir = os.path.join(logdir, "head_n={}-mixer={}-hpn_hyperdim={}-acti={}".format(
                args.hpn_head_num, args.mixer, args.hpn_hyper_dim, args.hpn_hyper_activation))

    logdir = os.path.join(logdir, "rnn_dim={}-2bs={}_{}-tdlambda={}-epdec_{}={}k".format(
        args.rnn_hidden_dim, args.buffer_size, args.batch_size,
        args.td_lambda, args.epsilon_finish, args.epsilon_anneal_time // 1000))
    args.log_model_dir = logdir
    
    # Setup tensorboard (main process only)
    if is_main and args.use_tensorboard:
        tb_logs_direc = os.path.join(dirname(dirname(dirname(abspath(__file__)))), args.local_results_path, "tb_logs")
        tb_exp_direc = os.path.join(tb_logs_direc, "{}").format(unique_token)
        if args.name in testing_algorithms:
            tb_exp_direc = os.path.join(tb_logs_direc, logdir, unique_token)
        logger.setup_tb(tb_exp_direc)
    
    if is_main:
        logger.setup_sacred(_run)
    
    # Setup wandb (main process only)
    if is_main and getattr(args, 'use_wandb', False):
        wandb_config = _config.copy()
        map_name = args.env_args.get('map_name', 'unknown')
        world_size = get_world_size()
        if getattr(args, 'use_graph_reconstruction', False):
            stage = getattr(args, 'recontructer_stage', 'stage1')
            project_name = f"{map_name}_reconstruction_{stage}_ddp{world_size}"
        else:
            project_name = f"{map_name}_exp_ddp{world_size}"
        wandb_kwargs = {'project': project_name, 'name': unique_token, 'config': wandb_config}
        if hasattr(args, 'wandb_entity') and args.wandb_entity:
            wandb_kwargs['entity'] = args.wandb_entity
        logger.setup_wandb(**wandb_kwargs)
    
    # Run training
    run_sequential_ddp(args=args, logger=logger, rank=rank)
    
    if is_main:
        print("Exiting Main")
        print("Stopping all threads")
        for t in threading.enumerate():
            if t.name != "MainThread":
                print("Thread {} is alive! Is daemon: {}".format(t.name, t.daemon))
                t.join(timeout=1)
                print("Thread joined")
        print("Exiting script")


def run_sequential_ddp(args, logger, rank=0):
    """
    Sequential training loop with DDP support.
    Each GPU runs its own environment and shares gradients.
    """
    is_main = is_main_process()
    world_size = get_world_size()
    
    # Each process creates its own runner
    runner = r_REGISTRY[args.runner](args=args, logger=logger)
    
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
        args.state_ally_feats_size = env_info["state_ally_feats_size"]
        args.state_enemy_feats_size = env_info["state_enemy_feats_size"]
        args.obs_component = env_info["obs_component"]
        args.state_component = env_info["state_component"]
        args.map_type = env_info["map_type"]
        args.agent_own_state_size = env_info["state_ally_feats_size"]
    
    scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        "full_obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.int},
        "probs": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.float},
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
    }
    groups = {"agents": args.n_agents}
    preprocess = {"actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])}
    
    # Each GPU has its own buffer (samples are different per GPU due to environment randomness)
    buffer = ReplayBuffer(scheme, groups, args.buffer_size, env_info["episode_limit"] + 1,
                          preprocess=preprocess, device="cpu" if args.buffer_cpu_only else args.device)
    
    mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args)
    runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)
    
    # Use DDP learner
    learner_name = args.learner
    if not learner_name.endswith('_ddp'):
        learner_name = learner_name + '_ddp'
        if learner_name not in le_REGISTRY:
            learner_name = args.learner  # Fall back to non-DDP learner
    
    learner = le_REGISTRY[learner_name](mac, buffer.scheme, logger, args)
    
    if args.use_cuda:
        learner.cuda()
    
    # Synchronize initial model weights across all GPUs
    if dist.is_initialized():
        _sync_model_weights(learner)
    
    # Load checkpoints (on main process first, then broadcast)
    if args.checkpoint_path != "":
        _load_checkpoint(args, learner, logger, runner)
    
    # Load pretrained models
    pretrained_tokenizer_path = getattr(args, 'pretrained_tokenizer_path', '')
    pretrained_mask_predictor_path = getattr(args, 'pretrained_mask_predictor_path', '')
    
    if pretrained_tokenizer_path and is_main:
        logger.console_logger.info(f"Loading tokenizer from {pretrained_tokenizer_path}")
        learner.load_graph_reconstructor(pretrained_tokenizer_path, stage='stage1')
    
    if pretrained_mask_predictor_path and is_main:
        logger.console_logger.info(f"Loading mask predictor from {pretrained_mask_predictor_path}")
        learner.load_graph_reconstructor(pretrained_mask_predictor_path, stage='stage2')
    
    # Sync weights after loading pretrained models
    if dist.is_initialized():
        dist.barrier()
    
    # Check mode
    if getattr(args, 'evaluation_only', False):
        if is_main:
            from run.run_v2 import evaluation_only_graph_reconstructer
            evaluation_only_graph_reconstructer(args, runner, learner, logger)
        runner.close_env()
        return
    elif getattr(args, 'pretrain_only', True):
        # Pretrain mode
        pretrain_graph_reconstructer_ddp(args, runner, learner, buffer, logger)
        runner.close_env()
        if is_main:
            logger.console_logger.info("Pretrain completed. Set pretrain_only=False to run Stage 3.")
        sys.stdout.flush()
        time.sleep(10)
        return
    
    # Stage 3: QMIX Training
    if is_main:
        logger.console_logger.info("=" * 60)
        logger.console_logger.info(f"=== Stage 3: QMIX Training with DDP ({world_size} GPUs) ===")
        logger.console_logger.info("=" * 60)
    
    # Setup graph reconstruction for MAC
    if hasattr(mac, 'set_graph_reconstructer'):
        mac.set_graph_reconstructer(learner.graph_reconstructer)
        learner.target_mac.set_graph_reconstructer(learner.graph_reconstructer)
        if is_main:
            logger.console_logger.info("[Stage 3] Graph reconstructer enabled for MAC and target_MAC")
    
    # Freeze graph reconstructer
    freeze_graph = getattr(args, 'recontructer_stage', 'stage0') == 'stage3'
    if freeze_graph:
        learner.graph_reconstructer.eval()
        for param in learner.graph_reconstructer.parameters():
            param.requires_grad = False
        if is_main:
            logger.console_logger.info("[Stage 3] Graph reconstructer frozen")
    
    # Training loop
    episode = 0
    last_test_T = -args.test_interval - 1
    last_log_T = 0
    model_save_time = 0
    
    start_time = time.time()
    last_time = start_time
    
    if is_main:
        logger.console_logger.info("Beginning training for {} timesteps".format(args.t_max))
    
    while runner.t_env <= args.t_max:
        # Run episode
        with th.no_grad():
            episode_batch = runner.run(test_mode=False)
            if episode_batch.batch_size > 0:
                buffer.insert_episode_batch(episode_batch)
            episode += args.batch_size_run
        
        if buffer.can_sample(args.batch_size):
            if args.accumulated_episodes and episode % args.accumulated_episodes != 0:
                continue
            
            episode_sample = buffer.sample(args.batch_size)
            max_ep_t = episode_sample.max_t_filled()
            episode_sample = episode_sample[:, :max_ep_t]
            
            if episode_sample.device != args.device:
                episode_sample.to(args.device)
            
            learner.train(episode_sample, runner.t_env, episode)
            del episode_sample
        
        # Test runs (all processes run tests, but only main logs)
        n_test_runs = max(1, args.test_nepisode // runner.batch_size)
        if (runner.t_env - last_test_T) / args.test_interval >= 1.0:
            if is_main:
                logger.console_logger.info("t_env: {} / {}".format(runner.t_env, args.t_max))
                logger.console_logger.info("Estimated time left: {}. Time passed: {}".format(
                    time_left(last_time, last_test_T, runner.t_env, args.t_max), time_str(time.time() - start_time)))
            last_time = time.time()
            last_test_T = runner.t_env
            with th.no_grad():
                for _ in range(n_test_runs):
                    runner.run(test_mode=True)
        
        # Save model (main process only)
        if args.save_model and is_main and (
                runner.t_env - model_save_time >= args.save_model_interval or runner.t_env >= args.t_max):
            model_save_time = runner.t_env
            save_path = os.path.join(args.local_results_path, "models", args.log_model_dir, args.unique_token,
                                    str(runner.t_env))
            os.makedirs(save_path, exist_ok=True)
            logger.console_logger.info("Saving models to {}".format(save_path))
            learner.save_models(save_path)
        
        # Log stats (main process only)
        if is_main and (runner.t_env - last_log_T) >= args.log_interval:
            logger.log_stat("episode", episode, runner.t_env)
            logger.log_stat("episode_in_buffer", buffer.episodes_in_buffer, runner.t_env)
            logger.print_recent_stats()
            last_log_T = runner.t_env
    
    runner.close_env()
    if is_main:
        logger.console_logger.info("Finished Training")
    sys.stdout.flush()
    time.sleep(10)


def pretrain_graph_reconstructer_ddp(args, runner, learner, buffer, logger):
    """
    Pretrain graph reconstructer with DDP support.
    Each GPU trains on its own sampled batch, gradients are synchronized.
    """
    is_main = is_main_process()
    world_size = get_world_size()
    
    stage = getattr(args, 'recontructer_stage', 'stage1')
    pretrain_episodes = getattr(args, 'graph_pretrain_episodes', args.buffer_size)
    pretrain_epochs = getattr(args, 'graph_pretrain_epochs', 100)
    pretrain_batch_size = getattr(args, 'graph_pretrain_batch_size', args.batch_size)
    log_interval = getattr(args, 'graph_pretrain_log_interval', 10)
    eval_interval = getattr(args, 'graph_pretrain_eval_interval', 20)
    eval_episodes = getattr(args, 'graph_pretrain_eval_episodes', args.batch_size_run * 5)
    
    if is_main:
        logger.console_logger.info("=" * 60)
        logger.console_logger.info(f"=== Graph Reconstructer Pretrain: {stage} (DDP {world_size} GPUs) ===")
        logger.console_logger.info(f"    Episodes to collect: {pretrain_episodes}")
        logger.console_logger.info(f"    Training epochs: {pretrain_epochs}")
        logger.console_logger.info(f"    Batch size per GPU: {pretrain_batch_size}")
        logger.console_logger.info(f"    Effective batch size: {pretrain_batch_size * world_size}")
        logger.console_logger.info("=" * 60)
    
    # Phase 1: Collect training episodes (each GPU collects independently)
    if is_main:
        logger.console_logger.info(f"[Pretrain] Each GPU collecting {pretrain_episodes} training episodes...")
    
    with tqdm(total=pretrain_episodes, initial=buffer.episodes_in_buffer, 
              desc=f"[GPU {get_rank()}] Collecting", disable=not is_main) as pbar:
        while buffer.episodes_in_buffer < pretrain_episodes:
            with th.no_grad():
                episode_batch = runner.run(test_mode=False, skip_logging=True)
                if episode_batch.batch_size > 0:
                    buffer.insert_episode_batch(episode_batch)
                    pbar.update(episode_batch.batch_size)
    
    if is_main:
        logger.console_logger.info(f"[Pretrain] Training data collection complete")
    
    # Synchronize before training
    if dist.is_initialized():
        dist.barrier()
    
    best_eval_loss = float('inf')
    best_model_path = None
    
    # Phase 2: Train with DDP
    if is_main:
        logger.console_logger.info(f"[Pretrain] Training {stage} for {pretrain_epochs} epochs...")
    
    for epoch in range(pretrain_epochs):
        if not buffer.can_sample(pretrain_batch_size):
            continue
        
        episode_sample = buffer.sample(pretrain_batch_size)
        max_ep_t = episode_sample.max_t_filled()
        episode_sample = episode_sample[:, :max_ep_t]
        if episode_sample.device != args.device:
            episode_sample.to(args.device)
        
        loss_info = learner.train_graph_reconstructor(episode_sample, runner.t_env, epoch)
        
        # Evaluation (only main process evaluates, others wait)
        if (epoch + 1) % eval_interval == 0 or epoch == pretrain_epochs - 1:
            if dist.is_initialized():
                dist.barrier()
            
            if is_main:
                eval_loss, _ = _run_graph_evaluation_simple(args, runner, learner, logger, eval_episodes, epoch)
                
                if eval_loss < best_eval_loss and args.save_model:
                    best_eval_loss = eval_loss
                    best_model_path = os.path.join(
                        args.local_results_path, "models", args.log_model_dir,
                        args.unique_token, f"pretrain_{stage}_best"
                    )
                    os.makedirs(best_model_path, exist_ok=True)
                    learner.save_graph_reconstructor(best_model_path)
                    logger.console_logger.info(f"[Pretrain] New best model saved (loss: {eval_loss:.4f})")
            
            if dist.is_initialized():
                dist.barrier()
    
    if is_main:
        logger.console_logger.info(f"[Pretrain] {stage} training completed!")
        
        if args.save_model:
            save_path = os.path.join(
                args.local_results_path, "models", args.log_model_dir,
                args.unique_token, f"pretrain_{stage}"
            )
            os.makedirs(save_path, exist_ok=True)
            learner.save_graph_reconstructor(save_path)
            logger.console_logger.info(f"[Pretrain] Saved final {stage} model to {save_path}")


def _run_graph_evaluation_simple(args, runner, learner, logger, eval_episodes, epoch):
    """Simple evaluation for graph reconstructer."""
    batches = []
    collected = 0
    while collected < eval_episodes:
        with th.no_grad():
            episode_batch = runner.run(test_mode=True, skip_logging=True)
            if episode_batch.batch_size > 0:
                batches.append(episode_batch)
                collected += episode_batch.batch_size
    
    total_loss = 0.0
    for batch in batches:
        max_ep_t = batch.max_t_filled()
        batch = batch[:, :max_ep_t]
        if batch.device != args.device:
            batch = batch.to(args.device)
        loss, _ = learner.eval_graph_reconstructor(batch)
        total_loss += loss
    
    avg_loss = total_loss / max(len(batches), 1)
    logger.console_logger.info(f"[Eval] Epoch {epoch} - Loss: {avg_loss:.4f}")
    logger.log_stat("eval/graph_loss", avg_loss, epoch)
    
    return avg_loss, {}


def _sync_model_weights(learner):
    """Synchronize model weights across all processes from rank 0."""
    # Broadcast agent weights
    if hasattr(learner.mac, 'agent'):
        for param in learner.mac.agent.parameters():
            dist.broadcast(param.data, src=0)
    
    # Broadcast mixer weights
    if learner.mixer is not None:
        from utils.dist_utils import unwrap_model
        mixer = unwrap_model(learner.mixer)
        for param in mixer.parameters():
            dist.broadcast(param.data, src=0)
    
    # Broadcast graph reconstructer weights if exists
    if hasattr(learner, 'graph_reconstructer'):
        for param in learner.graph_reconstructer.parameters():
            dist.broadcast(param.data, src=0)


def _load_checkpoint(args, learner, logger, runner):
    """Load checkpoint (handles DDP case)."""
    is_main = is_main_process()
    
    timesteps = []
    if not os.path.isdir(args.checkpoint_path):
        if is_main:
            logger.console_logger.info("Checkpoint directory {} doesn't exist".format(args.checkpoint_path))
        return
    
    for name in os.listdir(args.checkpoint_path):
        full_name = os.path.join(args.checkpoint_path, name)
        if os.path.isdir(full_name) and name.isdigit():
            timesteps.append(int(name))
    
    if not timesteps:
        return
    
    if args.load_step == 0:
        timestep_to_load = max(timesteps)
    else:
        timestep_to_load = min(timesteps, key=lambda x: abs(x - args.load_step))
    
    model_path = os.path.join(args.checkpoint_path, str(timestep_to_load))
    if is_main:
        logger.console_logger.info("Loading model from {}".format(model_path))
    learner.load_models(model_path)
    runner.t_env = timestep_to_load


def args_sanity_check(config, _log):
    if config["use_cuda"] and not th.cuda.is_available():
        config["use_cuda"] = False
        _log.warning("CUDA flag use_cuda was switched OFF automatically because no CUDA devices are available!")
    
    if config["test_nepisode"] < config["batch_size_run"]:
        config["test_nepisode"] = config["batch_size_run"]
    else:
        config["test_nepisode"] = (config["test_nepisode"] // config["batch_size_run"]) * config["batch_size_run"]
    
    return config
