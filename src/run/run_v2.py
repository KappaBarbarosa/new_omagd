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
from collections import Counter

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

    args.device = "cuda" if args.use_cuda else "cpu"

    # setup loggers
    logger = Logger(_log)

    _log.info("Experiment Parameters:")
    experiment_params = pprint.pformat(_config, indent=4, width=1)
    _log.info("\n\n" + experiment_params + "\n")

    # configure tensorboard logger
    unique_token = "{}__{}".format(args.name, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    args.unique_token = unique_token

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
    
    if args.use_tensorboard:
        tb_logs_direc = os.path.join(dirname(dirname(dirname(abspath(__file__)))), args.local_results_path, "tb_logs")
        tb_exp_direc = os.path.join(tb_logs_direc, "{}").format(unique_token)
        if args.name in testing_algorithms:
            tb_exp_direc = os.path.join(tb_logs_direc, logdir, unique_token)
        logger.setup_tb(tb_exp_direc)

    logger.setup_sacred(_run)

    if getattr(args, 'use_wandb', False):
        wandb_config = _config.copy()
        if getattr(args, 'use_graph_reconstruction', False):
            map_name = args.env_args.get('map_name', 'unknown')
            stage = getattr(args, 'recontructer_stage', 'stage1')
            project_name = f"{map_name}_reconstruction_{stage}"
        else:
            project_name = f"{map_name}_exp"
        wandb_kwargs = {'project': project_name, 'name': unique_token, 'config': wandb_config}
        if hasattr(args, 'wandb_entity') and args.wandb_entity:
            wandb_kwargs['entity'] = args.wandb_entity
        logger.setup_wandb(**wandb_kwargs)

    run_sequential(args=args, logger=logger)

    print("Exiting Main")
    print("Stopping all threads")
    for t in threading.enumerate():
        if t.name != "MainThread":
            print("Thread {} is alive! Is daemon: {}".format(t.name, t.daemon))
            t.join(timeout=1)
            print("Thread joined")
    print("Exiting script")
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
    pretrain_buffer_path = getattr(args, 'pretrain_buffer_path', '')
    save_pretrain_buffer = getattr(args, 'save_pretrain_buffer', False)
    
    logger.console_logger.info("=" * 60)
    logger.console_logger.info(f"=== Graph Reconstructer Pretrain: {stage} ===")
    logger.console_logger.info(f"    Episodes to collect: {pretrain_episodes}")
    logger.console_logger.info(f"    Training epochs: {pretrain_epochs}")
    logger.console_logger.info(f"    Batch size: {pretrain_batch_size}")
    logger.console_logger.info(f"    Eval interval: {eval_interval} epochs")
    logger.console_logger.info(f"    Eval episodes: {eval_episodes}")
    if pretrain_buffer_path:
        logger.console_logger.info(f"    Buffer load path: {pretrain_buffer_path}")
    if save_pretrain_buffer:
        logger.console_logger.info(f"    Save buffer after collection: True")
    logger.console_logger.info("=" * 60)
    
    # Phase 1: Load or collect training episodes
    buffer_loaded = False
    if pretrain_buffer_path and os.path.exists(pretrain_buffer_path):
        logger.console_logger.info(f"[Pretrain] Loading episodes from {pretrain_buffer_path}...")
        buffer_loaded = buffer.load(pretrain_buffer_path, strict=True)
        if buffer_loaded:
            logger.console_logger.info(f"[Pretrain] Successfully loaded {buffer.episodes_in_buffer} episodes from disk")
        else:
            logger.console_logger.warning(f"[Pretrain] Failed to load buffer, will collect new episodes")
    
    if not buffer_loaded:
        logger.console_logger.info(f"[Pretrain] Collecting {pretrain_episodes} training episodes...")
        with tqdm(total=pretrain_episodes, initial=buffer.episodes_in_buffer, desc="Collecting pretrain episodes") as pbar:
            while buffer.episodes_in_buffer < pretrain_episodes:
                with th.no_grad():
                    episode_batch = runner.run(test_mode=False, skip_logging=True)
                    if episode_batch.batch_size > 0:
                        buffer.insert_episode_batch(episode_batch)
                        pbar.update(episode_batch.batch_size)
        logger.console_logger.info(f"[Pretrain] Training data collection complete: {buffer.episodes_in_buffer} episodes")
        
        if save_pretrain_buffer:
            save_dir = os.path.join(args.local_results_path, "buffers", args.log_model_dir)
            os.makedirs(save_dir, exist_ok=True)
            map_name = args.env_args.get('map_name', 'unknown')
            save_path = os.path.join(save_dir, f"pretrain_buffer_{map_name}_{pretrain_episodes}.pt")
            buffer.save(save_path)
            logger.console_logger.info(f"[Pretrain] Buffer saved to {save_path}")
    
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
        
        loss_info = learner.train_graph_reconstructor(episode_sample, runner.t_env, epoch)
        
        if epoch % log_interval == 0:
            _log_token_stats(logger, loss_info, epoch, prefix="[Train]")
        
        if (epoch + 1) % eval_interval == 0 or epoch == pretrain_epochs - 1:
            eval_loss, _ = _run_graph_evaluation(
                args, runner, learner, logger, eval_episodes, epoch=epoch,
                detailed_episodes=getattr(args, 'eval_detailed_episodes', 3)
            )
            
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


# ==============================================================================
# Shared Evaluation Helpers
# ==============================================================================

def _collect_episodes(runner, num_episodes, test_mode=True, desc="Collecting episodes"):
    """Collect episodes from the runner."""
    batches = []
    collected = 0
    with tqdm(total=num_episodes, desc=desc) as pbar:
        while collected < num_episodes:
            with th.no_grad():
                episode_batch = runner.run(test_mode=test_mode, skip_logging=True)
                if episode_batch.batch_size > 0:
                    batches.append(episode_batch)
                    collected += episode_batch.batch_size
                    pbar.update(episode_batch.batch_size)
    return batches


def _compute_aggregate_metrics(args, learner, eval_batches):
    """Compute aggregate evaluation metrics from collected batches."""
    total_loss = 0.0
    all_metrics = {}
    all_gt_tokens, all_pred_tokens = [], []
    total_masked_count = 0
    num_batches = len(eval_batches)
    
    with th.no_grad():
        for batch in eval_batches:
            max_ep_t = batch.max_t_filled()
            batch = batch[:, :max_ep_t]
            if batch.device != args.device:
                batch = batch.to(args.device)
            
            loss, metrics = learner.eval_graph_reconstructor(batch)
            total_loss += loss
            
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    all_metrics[key] = all_metrics.get(key, 0.0) + value
            
            if 'masked_gt_top5_tokens' in metrics:
                for t, c in zip(metrics.get('masked_gt_top5_tokens', []), metrics.get('masked_gt_top5_counts', [])):
                    all_gt_tokens.append((t, c))
                for t, c in zip(metrics.get('masked_pred_top5_tokens', []), metrics.get('masked_pred_top5_counts', [])):
                    all_pred_tokens.append((t, c))
                total_masked_count += metrics.get('masked_gt_total_count', 0)
    
    gt_counter, pred_counter = Counter(), Counter()
    for token, count in all_gt_tokens:
        gt_counter[token] += count
    for token, count in all_pred_tokens:
        pred_counter[token] += count
    
    token_stats = {
        'gt_top10': gt_counter.most_common(10),
        'pred_top10': pred_counter.most_common(10),
        'total_masked': total_masked_count,
        'unique_gt_tokens': len(gt_counter),
    }
    
    avg_loss = total_loss / max(num_batches, 1)
    avg_metrics = {k: v / max(num_batches, 1) for k, v in all_metrics.items()}
    return avg_loss, avg_metrics, token_stats


def _log_token_stats(logger, info, epoch, prefix="[Eval]"):
    """Log masked token statistics."""
    if 'masked_gt_top5_tokens' not in info:
        return
    gt_tokens = info.get('masked_gt_top5_tokens', [])
    gt_counts = info.get('masked_gt_top5_counts', [])
    pred_tokens = info.get('masked_pred_top5_tokens', [])
    pred_counts = info.get('masked_pred_top5_counts', [])
    total = info.get('masked_gt_total_count', 0)
    unique = info.get('masked_gt_unique_count', 0)
    
    gt_str = ", ".join([f"{t}({c})" for t, c in zip(gt_tokens, gt_counts)])
    pred_str = ", ".join([f"{t}({c})" for t, c in zip(pred_tokens, pred_counts)])
    
    logger.console_logger.info(f"{prefix} Epoch {epoch} - Masked tokens: total={total}, unique_gt={unique}")
    logger.console_logger.info(f"{prefix} Epoch {epoch} - GT token dist (top5): [{gt_str}]")
    logger.console_logger.info(f"{prefix} Epoch {epoch} - Pred token dist (top5): [{pred_str}]")


def _log_evaluation_results(args, logger, avg_loss, avg_metrics, token_stats, epoch=None, prefix="[Eval]"):
    """Log evaluation results to console."""
    stage = getattr(args, 'recontructer_stage', 'stage1')
    
    if epoch is not None:
        logger.console_logger.info(f"{prefix} Epoch {epoch} - Eval Loss: {avg_loss:.4f}")
        logger.log_stat("eval/graph_loss", avg_loss, epoch)
        for key, value in avg_metrics.items():
            if isinstance(value, (int, float)):
                logger.log_stat(f"eval/_{key}", value, epoch)
    else:
        logger.console_logger.info(f"{prefix} Average Loss: {avg_loss:.4f}")
    
    if token_stats.get('gt_top10'):
        gt_str = ", ".join([f"{t}({c})" for t, c in token_stats['gt_top10']])
        pred_str = ", ".join([f"{t}({c})" for t, c in token_stats['pred_top10']])
        logger.console_logger.info(f"{prefix} Masked tokens: total={token_stats['total_masked']}, unique_gt={token_stats['unique_gt_tokens']}")
        logger.console_logger.info(f"{prefix} GT token dist (top10): [{gt_str}]")
        logger.console_logger.info(f"{prefix} Pred token dist (top10): [{pred_str}]")
    
    if avg_metrics:
        if stage == 'stage1':
            display_keys = ['mse', 'cosine_similarity', 'feature_correlation', 'perplexity', 'codebook_usage']
        else:
            display_keys = ['masked_accuracy', 'top1_accuracy', 'top3_accuracy', 'top5_accuracy',
                           'mrr', 'hungarian_accuracy', 'predicted_mse', 'gt_mse', 'predicted_cosim', 'gt_cosim']
        metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in avg_metrics.items() if k in display_keys])
        if metrics_str:
            logger.console_logger.info(f"{prefix} Metrics: {metrics_str}")


def _log_sample_comparisons(args, avg_metrics, logger, prefix="[Eval]"):
    """Display sample comparisons based on stage."""
    stage = getattr(args, 'recontructer_stage', 'stage1')
    if stage == 'stage1' and 'samples' in avg_metrics:
        from modules.graph_reconstructers.tokenizer_logger import format_sample_comparison
        sample_str = format_sample_comparison(avg_metrics.get('samples', []))
        if sample_str:
            logger.console_logger.info(f"{prefix} Sample Reconstructions:\n{sample_str}")
    elif stage == 'stage2' and 'token_samples' in avg_metrics:
        from modules.graph_reconstructers.mask_predictor_logger import format_token_samples
        sample_str = format_token_samples(avg_metrics.get('token_samples', []), metrics=avg_metrics)
        if sample_str:
            logger.console_logger.info(f"{prefix} Token Samples:\n{sample_str}")


def _run_detailed_episode_logging(args, runner, learner, logger, detailed_episodes):
    """Collect and print detailed step-by-step episode logs."""
    if detailed_episodes <= 0:
        return
    
    mask_pred_cfg = getattr(args, 'mask_predictor_config', {})
    stacked_steps = mask_pred_cfg.get('stacked_steps', 1)
    stacked_strip = mask_pred_cfg.get('stacked_strip', 1)
    
    logger.console_logger.info(f"[Eval] Collecting {detailed_episodes} episode(s) for detailed logging...")
    
    detailed_batches = []
    collected = 0
    while collected < detailed_episodes:
        with th.no_grad():
            episode_batch = runner.run(test_mode=True, skip_logging=True)
            if episode_batch.batch_size > 0:
                for i in range(min(episode_batch.batch_size, detailed_episodes - collected)):
                    detailed_batches.append(episode_batch[i:i+1])
                    collected += 1
                    if collected >= detailed_episodes:
                        break
    
    if stacked_steps > 1:
        from modules.graph_reconstructers.evaluation.episode_logger import print_detailed_episode_stacked
        log_func = print_detailed_episode_stacked
    else:
        from modules.graph_reconstructers.evaluation.episode_logger import print_detailed_episode_full
        log_func = print_detailed_episode_full
    
    for i, batch in enumerate(detailed_batches):
        max_ep_t = batch.max_t_filled()
        batch = batch[:, :max_ep_t]
        if batch.device != args.device:
            batch = batch.to(args.device)
        log_func(
            batch=batch,
            graph_reconstructer=learner.graph_reconstructer,
            stacked_steps=stacked_steps,
            # stacked_strip=stacked_strip if stacked_steps > 1 else None,
            n_nodes_per_frame=learner.graph_reconstructer.n_nodes_per_frame,
            episode_num=i + 1,
            agent_idx=0,
            logger=logger.console_logger,
        )


def _run_graph_evaluation(args, runner, learner, logger, eval_episodes, epoch=None, detailed_episodes=0):
    """
    Core evaluation logic shared by evaluate_graph_reconstructer and evaluation_only.
    
    Returns:
        Tuple of (avg_loss, avg_metrics)
    """
    logger.console_logger.info(f"[Eval] Collecting {eval_episodes} episodes for evaluation...")
    eval_batches = _collect_episodes(runner, eval_episodes, test_mode=True, desc="Collecting eval episodes")
    
    avg_loss, avg_metrics, token_stats = _compute_aggregate_metrics(args, learner, eval_batches)
    _log_evaluation_results(args, logger, avg_loss, avg_metrics, token_stats, epoch=epoch, prefix="[Eval]")
    _log_sample_comparisons(args, avg_metrics, logger, prefix="[Eval]")
    
    if detailed_episodes > 0:
        _run_detailed_episode_logging(args, runner, learner, logger, detailed_episodes)
    
    return avg_loss, avg_metrics


# ==============================================================================
# Main Evaluation Functions
# ==============================================================================

def evaluate_graph_reconstructer(args, runner, learner, logger, eval_episodes, epoch):
    """Evaluate graph reconstructer during training."""
    detailed_episodes = getattr(args, 'eval_detailed_episodes', 3)
    return _run_graph_evaluation(args, runner, learner, logger, eval_episodes, epoch=epoch, detailed_episodes=detailed_episodes)


def evaluation_only_graph_reconstructer(args, runner, learner, logger):
    """Evaluation-only mode for graph reconstructer."""
    stage = getattr(args, 'recontructer_stage', 'stage1')
    eval_episodes = getattr(args, 'eval_only_episodes', getattr(args, 'graph_pretrain_eval_episodes', 40))
    detailed_episodes = getattr(args, 'eval_report_episodes', 3)
    mask_pred_cfg = getattr(args, 'mask_predictor_config', {})
    stacked_steps = mask_pred_cfg.get('stacked_steps', 1)
    
    logger.console_logger.info("=" * 80)
    logger.console_logger.info(f"=== Evaluation-Only Mode: {stage} ===")
    logger.console_logger.info(f"    Aggregate evaluation episodes: {eval_episodes}")
    logger.console_logger.info(f"    Detailed logging episodes: {detailed_episodes}")
    logger.console_logger.info(f"    Stacked steps: {stacked_steps}")
    logger.console_logger.info("=" * 80)
    
    # Ensure model is on correct device and properly configured
    if args.use_cuda:
        learner.graph_reconstructer = learner.graph_reconstructer.to(args.device)
    
    # Use set_training_stage to properly freeze tokenizer for stage2 (same as training eval)
    # This ensures consistent token generation between training eval and eval_only
    learner.graph_reconstructer.set_training_stage(stage)
    learner.graph_reconstructer.eval()
    
    logger.console_logger.info(f"    Tokenizer frozen: {stage == 'stage2'}")
    
    logger.console_logger.info(f"\n[Phase 1] Aggregate evaluation ({eval_episodes} episodes)...")
    _run_graph_evaluation(args, runner, learner, logger, eval_episodes, epoch=None, detailed_episodes=0)
    
    logger.console_logger.info(f"\n[Phase 2] Detailed logging ({detailed_episodes} episodes)...")
    _run_detailed_episode_logging(args, runner, learner, logger, detailed_episodes)
    
    logger.console_logger.info("=" * 80)
    logger.console_logger.info("Evaluation-Only Mode Complete!")
    logger.console_logger.info("=" * 80)


def run_sequential(args, logger):
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
    
    buffer = ReplayBuffer(scheme, groups, args.buffer_size, env_info["episode_limit"] + 1,
                          preprocess=preprocess, device="cpu" if args.buffer_cpu_only else args.device)
    mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args)
    runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)
    learner = le_REGISTRY[args.learner](mac, buffer.scheme, logger, args)

    if args.use_cuda:
        learner.cuda()

    if args.checkpoint_path != "":
        timesteps = []
        if not os.path.isdir(args.checkpoint_path):
            logger.console_logger.info("Checkpoint directory {} doesn't exist".format(args.checkpoint_path))
            return

        for name in os.listdir(args.checkpoint_path):
            full_name = os.path.join(args.checkpoint_path, name)
            if os.path.isdir(full_name) and name.isdigit():
                timesteps.append(int(name))

        if args.load_step == 0:
            timestep_to_load = max(timesteps)
        else:
            timestep_to_load = min(timesteps, key=lambda x: abs(x - args.load_step))

        model_path = os.path.join(args.checkpoint_path, str(timestep_to_load))
        logger.console_logger.info("Loading model from {}".format(model_path))
        learner.load_models(model_path)
        runner.t_env = timestep_to_load

        if args.evaluate or args.save_replay:
            evaluate_sequential(args, runner)
            return

    pretrained_tokenizer_path = getattr(args, 'pretrained_tokenizer_path', '')
    pretrained_mask_predictor_path = getattr(args, 'pretrained_mask_predictor_path', '')
    
    if pretrained_tokenizer_path:
        logger.console_logger.info(f"Loading tokenizer from {pretrained_tokenizer_path}")
        learner.load_graph_reconstructor(pretrained_tokenizer_path, stage='stage1')
    
    if pretrained_mask_predictor_path:
        logger.console_logger.info(f"Loading mask predictor from {pretrained_mask_predictor_path}")
        learner.load_graph_reconstructor(pretrained_mask_predictor_path, stage='stage2')

    if getattr(args, 'evaluation_only', False):
        evaluation_only_graph_reconstructer(args, runner, learner, logger)
        runner.close_env()
        return
    elif getattr(args, 'pretrain_only', True):
        # Stage 1/2 pretrain only (default behavior)
        pretrain_graph_reconstructer(args, runner, learner, buffer, logger)
        runner.close_env()
        logger.console_logger.info("Pretrain completed. Set pretrain_only=False to run Stage 3.")
        sys.stdout.flush()
        time.sleep(10)
        return

    # ========== Stage 3: QMIX Training with Reconstructed Observations ==========
    logger.console_logger.info("=" * 60)
    logger.console_logger.info("=== Stage 3: QMIX Training with Graph Reconstruction ===")
    logger.console_logger.info("=" * 60)
    
    # Setup graph reconstruction for MAC (if using NGraphMAC)
    if hasattr(mac, 'set_graph_reconstructer'):
        mac.set_graph_reconstructer(learner.graph_reconstructer)
        logger.console_logger.info("[Stage 3] Graph reconstructer enabled for MAC")
    
    # Freeze graph reconstructer during Stage 3 (default)
    freeze_graph = getattr(args, 'recontructer_stage', 'stage0') == 'stage3'
    if freeze_graph:
        learner.graph_reconstructer.eval()
        for param in learner.graph_reconstructer.parameters():
            param.requires_grad = False
        logger.console_logger.info("[Stage 3] Graph reconstructer frozen")
    
    episode = 0
    last_test_T = -args.test_interval - 1
    last_log_T = 0
    model_save_time = 0

    start_time = time.time()
    last_time = start_time

    logger.console_logger.info("Beginning training for {} timesteps".format(args.t_max))

    while runner.t_env <= args.t_max:
        # Run for a whole episode at a time
        with th.no_grad():
            episode_batch = runner.run(test_mode=False)
            if episode_batch.batch_size > 0:
                buffer.insert_episode_batch(episode_batch)
            episode += args.batch_size_run

        if buffer.can_sample(args.batch_size):
            if args.accumulated_episodes and episode % args.accumulated_episodes != 0:
                continue

            episode_sample = buffer.sample(args.batch_size)

            # Truncate batch to only filled timesteps
            max_ep_t = episode_sample.max_t_filled()
            episode_sample = episode_sample[:, :max_ep_t]

            if episode_sample.device != args.device:
                episode_sample.to(args.device)

            learner.train(episode_sample, runner.t_env, episode)
            del episode_sample

        # Execute test runs once in a while
        n_test_runs = max(1, args.test_nepisode // runner.batch_size)
        if (runner.t_env - last_test_T) / args.test_interval >= 1.0:
            logger.console_logger.info("t_env: {} / {}".format(runner.t_env, args.t_max))
            logger.console_logger.info("Estimated time left: {}. Time passed: {}".format(
                time_left(last_time, last_test_T, runner.t_env, args.t_max), time_str(time.time() - start_time)))
            last_time = time.time()
            last_test_T = runner.t_env
            with th.no_grad():
                for _ in range(n_test_runs):
                    runner.run(test_mode=True)

        if args.save_model and (
                runner.t_env - model_save_time >= args.save_model_interval or runner.t_env >= args.t_max):
            model_save_time = runner.t_env
            save_path = os.path.join(args.local_results_path, "models", args.log_model_dir, args.unique_token,
                                     str(runner.t_env))
            os.makedirs(save_path, exist_ok=True)
            logger.console_logger.info("Saving models to {}".format(save_path))
            learner.save_models(save_path)

        if (runner.t_env - last_log_T) >= args.log_interval:
            logger.log_stat("episode", episode, runner.t_env)
            logger.log_stat("episode_in_buffer", buffer.episodes_in_buffer, runner.t_env)
            logger.print_recent_stats()
            last_log_T = runner.t_env

    runner.close_env()
    logger.console_logger.info("Finished Training")
    sys.stdout.flush()
    time.sleep(10)


def args_sanity_check(config, _log):
    if config["use_cuda"] and not th.cuda.is_available():
        config["use_cuda"] = False
        _log.warning("CUDA flag use_cuda was switched OFF automatically because no CUDA devices are available!")

    if config["test_nepisode"] < config["batch_size_run"]:
        config["test_nepisode"] = config["batch_size_run"]
    else:
        config["test_nepisode"] = (config["test_nepisode"] // config["batch_size_run"]) * config["batch_size_run"]

    return config
