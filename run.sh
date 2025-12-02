export WANDB_API_KEY=247e23f9da34555c8f9d172474c4d49ad150e88d
export CUDA_VISIBLE_DEVICES=0
python src/main.py --config=hpn_qmix --env-config=sc2 with env_args.map_name=5m_vs_6m obs_agent_id=True obs_last_action=False runner=parallel batch_size_run=8 buffer_size=5000 t_max=10050000 epsilon_anneal_time=100000 batch_size=128 td_lambda=0.6