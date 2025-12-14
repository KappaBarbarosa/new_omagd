export CUDA_VISIBLE_DEVICES=0
export WANDB_API_KEY=247e23f9da34555c8f9d172474c4d49ad150e88d
# Stage 1: 訓練 Tokenizer
python src/main.py --config=omagd --env-config=sc2 \
  with env_args.map_name=3m recontructer_stage=stage1 \
  use_wandb=True

# Stage 2: 訓練 Mask Predictor (載入 tokenizer)
# python src/main.py --config=omagd --env-config=sc2 \
#   with env_args.map_name=3m recontructer_stage=stage2 \
#   pretrained_tokenizer_path="results/models/.../pretrain_stage1"