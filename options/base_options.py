import argparse


class BaseOption:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('-d', '--data-root', type=str, default='./data/DVF_recons', help='the dataset root')
        self.parser.add_argument('--ckpt', type=str, default='./output/weights/model.pth', help='the checkpoint path')
        self.parser.add_argument('--mllm-ckpt', type=str, default='sparklexfantasy/llava-1.5-7b-rfrd', help='the checkpoint of mllm')
        self.parser.add_argument('--mllm-base', type=str, default=None, help='the base model of mllm')
        self.parser.add_argument('--st-ckpt', type=str, default='weights/ViT/vit_base_r50_s16_224.orig_in21k/jx_vit_base_resnet50_224_in21k-6f7c7740.pth', help='the checkpoint of the pretrained checkpoint of hybrid vit in ST branch')
        self.parser.add_argument('--st-pretrained', type=bool, default=True, help='whether to use the pretrained checkpoint of hybrid vit in ST branch')
        self.parser.add_argument('--model-name', type=str, default='MMDet', help='the model name')
        self.parser.add_argument('--expt', type=str, default='MMDet_01', help='the experiment name')
        self.parser.add_argument('--out-dir', type=str, default='outputs', help='the experiment name')
        self.parser.add_argument('--window-size', type=int, default=10, help='window size for video clips')
        self.parser.add_argument('--sample-frame', type=int, default=1, help='sampled frame numbers in a clip')
        self.parser.add_argument('--seed', type=int, default=9999, help='random seed')
        self.parser.add_argument('--gpus', type=int, default=1, help='number for gpus')

        
    def parse(self):
        return self.parser.parse_args()