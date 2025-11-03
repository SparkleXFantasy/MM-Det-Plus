CUDA_VISIBLE_DEVICES=0 python train.py \
  --data-root ./data/DVF_PP/ \
  --st-ckpt ./weights/vit_base_r50_s16_224.orig_in21k/jx_vit_base_resnet50_224_in21k-6f7c7740.pth \
  --mllm-ckpt ./weights/llava-1.5-7b-vctuned/ \