CUDA_VISIBLE_DEVICES=0 python eval.py \
  --data-root $YOUR_DATA_ROOT \
  --classes customized \
  --mllm-ckpt ./weights/llava-1.5-7b-vctuned/ \
  --ckpt-path ./weights/MMDetPlus/current_model.pth \