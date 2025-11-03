import torch
import os
import numpy as np
from functools import reduce
from PIL import Image
from torch import nn
from einops import rearrange
from transformers import BertTokenizer, CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig, AutoProcessor
from tqdm import tqdm
import torch.nn.functional as F

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model as load_llava_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, get_anyres_image_grid_shape

from .vit.stv_transformer_hybrid import stv_base_r50_s16_224_alpha


class CrossAttention(nn.Module):
    def __init__(self, embed_dim=1024, hidden_dim=512):
        super(CrossAttention, self).__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.linear_q = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.linear_k = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.linear_v = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.linear_output = nn.Linear(hidden_dim, embed_dim)
        self.fc = nn.Linear(embed_dim, embed_dim, bias=False)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.linear_q.weight)
        nn.init.xavier_uniform_(self.linear_k.weight)
        nn.init.xavier_uniform_(self.linear_v.weight)
        nn.init.xavier_uniform_(self.linear_output.weight)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.linear_output.bias)
    
    def forward(self, q_feat, kv_feat, attention_mask=None):
        batch_size, seq_len_q, _ = q_feat.size()
        _, seq_len_kv, _ = kv_feat.size()
        
        Q = self.linear_q(q_feat)  # [batch_size, seq_len_q, hidden_dim]
        K = self.linear_k(kv_feat)  # [batch_size, seq_len_kv, hidden_dim]
        V = self.linear_v(kv_feat)  # [batch_size, seq_len_kv, hidden_dim]
        
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.hidden_dim ** 0.5)
        if attention_mask is not None:
            attn_scores = attn_scores.masked_fill(attention_mask == 0, -1e9)
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        attn_output = torch.matmul(attn_weights, V)  # [batch_size, seq_len_q, hidden_dim]
        output = self.linear_output(attn_output)  # [batch_size, seq_len_q, embed_dim]
        
        output = output + q_feat
        output = self.fc(output)
        output = self.layer_norm(output)
        return output
    
    
def unpad_image(tensor, original_size):
    """
    Unpads a PyTorch tensor of a padded and resized image.

    Args:
    tensor (torch.Tensor): The image tensor, assumed to be in CxHxW format.
    original_size (tuple): The original size of the image (height, width).

    Returns:
    torch.Tensor: The unpadded image tensor.
    """
    original_width, original_height = original_size
    current_height, current_width = tensor.shape[1:]

    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    if original_aspect_ratio > current_aspect_ratio:
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[:, padding:current_height - padding, :]
    else:
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        unpadded_tensor = tensor[:, :, padding:current_width - padding]

    return unpadded_tensor

    
class LlavaForwardEncoder(nn.Module):
    def __init__(
        self,
        model_path='sparklexfantasy/llava-1.5-7b-vctuned',
        model_base=None,
        conv_mode='llava_v1',
        freezed=True,
        **kwargs
    ):
        super().__init__()
        self.model_path = model_path
        self.model_base = model_base
        self.model_name = get_model_name_from_path(model_path)
        self.conv_mode = conv_mode
        if 'mistral' in self.model_name:
            self.conv_mode = 'mistral'
        self.tokenizer, self.model, self.image_processor, self.context_len = load_llava_pretrained_model(model_path=model_path, model_base=model_base, model_name=self.model_name, load_4bit=True)
        vision_tower_config_path = getattr(self.get_vision_tower().config, "_name_or_path")
        self.vision_processor = AutoProcessor.from_pretrained(vision_tower_config_path)
        self.conv_mode = conv_mode
        self.llm_cls_token = nn.Parameter(torch.randn((1, self.model.config.hidden_size), dtype=torch.float16))
        self.freezed = freezed
        if self.freezed:
            for param in self.model.parameters():
                param.requires_grad = False
    
    def get_vision_tower(self):
        return self.model.get_vision_tower().vision_tower
        
    def get_prompt(self, embed_text=False):
        if self.conv_mode == 'llava_v1':
            if embed_text:
                assistant_intro = "Assistant: A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.###"
                human_instruction = "Human: <image>\nAs an expert in image forensics, you are to briefly analyze and describe the image. Give a reason to justify whether it is a real or a fake image.###"
                assistant_response = "Assistant:"
                text = assistant_intro + human_instruction + assistant_response
            else:
                text = None
        elif self.conv_mode == 'mistral':
            if embed_text:
                text = "[INST] <image>\nAs an expert in image forensics, you are to briefly analyze and describe the image. Give a reason to justify whether it is a real or a fake image. [/INST]"
            else:
                text = "[INST] [/INST]"
        return text
    
    def encode_vision_features(self, images, image_sizes=None):
        vision_input = self.vision_processor(images=images, return_tensors="pt").to(self.get_vision_tower().device)
        vision_input['pixel_values'] = vision_input['pixel_values'].half()
        vision_features = self.get_vision_tower()(**vision_input).pooler_output
        return vision_features
            
    def forward_explicit(self, images, embed_text=False):
        B = len(images)
        assert(B != 0)
        with torch.inference_mode():
            image_sizes = [image.size for image in images]
            image_t = process_images(images, self.image_processor, self.model.config)
            if type(image_t) is list:
                image_t = [image.to(self.model.device, dtype=torch.float16) for image in image_t]
            else:
                image_t = [image_t.to(self.model.device, dtype=torch.float16)]
            prompt = self.get_prompt(embed_text)
            if prompt is None:    # only images
                input_ids = None
            else:    # images + prompt                
                input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).repeat(B, 1).to(self.model.device)
            (
                _,
                _,
                _,
                _,
                inputs_embeds,
                _,
                vision_embeds
            ) = self.model.prepare_multimodal_embedding(
                input_ids,
                position_ids=None,
                attention_mask=None,
                past_key_values=None,
                labels=None,
                images=image_t,
                image_sizes=image_sizes
            )
            inputs_embeds = inputs_embeds.to(self.model.device)
            vision_embeds = vision_embeds.float().to(self.model.device)
            new_inputs_embeds = inputs_embeds
            new_inputs_embeds = torch.cat([inputs_embeds, self.llm_cls_token.to(inputs_embeds.device).unsqueeze(0).repeat(B, 1, 1)], dim=1).to(self.model.device)

            position_ids = torch.arange(
                0, new_inputs_embeds.size(1), dtype=torch.long, device=inputs_embeds.device
            )
            position_ids = position_ids.unsqueeze(0)
            attention_mask = None
            # attention_mask = torch.cat([torch.ones_like((B, 1), dtype=torch.bool), attention_mask], dim=1).to(self.model.device)
            # labels = torch.cat([torch.full((B, 1), IGNORE_INDEX), labels], dim=1).to(self.model.device)
            x = new_inputs_embeds.half()
            for layer in self.model.model.layers:
                out = layer(
                    x,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=None,
                    output_attentions=False,
                    use_cache=False,
                )
                x = out[0]
            x = self.model.model.norm(x)
            out_token_embeds = x[:, -1, :].float()
        return vision_embeds, out_token_embeds
    
    def forward_layer_explicit(self, images, embed_text=False, output_hidden_states=True):
        B = len(images)
        assert(B != 0)
        with torch.inference_mode():
            # if len(x.shape) == 4:
            #     x = x.unsqueeze(0)
            # B, L, C, H, W = x.shape
            # x = x[:, 0, :, :, :].squeeze(1)    # the first frame
            # x = rearrange(x, 'b c h w -> b h w c')
            # images = []
            # image_sizes = []
            # for t in x:
            #     img = Image.fromarray((t.cpu().numpy() * 255).astype(np.uint8))
            #     images.append(img)
            #     image_sizes.append(img.size)
            image_sizes = [image.size for image in images]
            image_t = process_images(images, self.image_processor, self.model.config)
            if type(image_t) is list:
                image_t = [image.to(self.model.device, dtype=torch.float16) for image in image_t]
            else:
                image_t = [image_t.to(self.model.device, dtype=torch.float16)]
            prompt = self.get_prompt(embed_text)
            if prompt is None:    # only images
                input_ids = None
            else:    # images + prompt                
                input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).repeat(B, 1).to(self.model.device)
            (
                _,
                _,
                _,
                _,
                inputs_embeds,
                _,
                vision_embeds
            ) = self.model.prepare_multimodal_embedding(
                input_ids,
                position_ids=None,
                attention_mask=None,
                past_key_values=None,
                labels=None,
                images=image_t,
                image_sizes=image_sizes
            )
            inputs_embeds = inputs_embeds.to(self.model.device)
            vision_embeds = vision_embeds.float().to(self.model.device)
            new_inputs_embeds = inputs_embeds
            new_inputs_embeds = torch.cat([inputs_embeds, self.llm_cls_token.to(inputs_embeds.device).unsqueeze(0).repeat(B, 1, 1)], dim=1).to(self.model.device)

            position_ids = torch.arange(
                0, new_inputs_embeds.size(1), dtype=torch.long, device=inputs_embeds.device
            )
            position_ids = position_ids.unsqueeze(0)
            attention_mask = None
            # attention_mask = torch.cat([torch.ones_like((B, 1), dtype=torch.bool), attention_mask], dim=1).to(self.model.device)
            # labels = torch.cat([torch.full((B, 1), IGNORE_INDEX), labels], dim=1).to(self.model.device)
            x = new_inputs_embeds.half()
            
            if output_hidden_states:
                hidden_states = [x]
            for layer in self.model.model.layers:
                out = layer(
                    x,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=None,
                    output_attentions=False,
                    use_cache=False,
                )
                x = out[0]
                if output_hidden_states:
                    hidden_states.append(x)
            x = self.model.model.norm(x)
            out_token_embeds = x[:, -1, :].float()
        if output_hidden_states:
            return vision_embeds, out_token_embeds, hidden_states
        else:
            return vision_embeds, out_token_embeds, hidden_states
    
    
class MMDetWUniModal(nn.Module):
    def __init__(
            self, 
            config,
            freezed_llm=True
        ):
        super(MMDetWUniModal, self).__init__()
        self.window_size = config['window_size']
        self.st_pretrained = config['st_pretrained']
        self.st_ckpt = config['st_ckpt']
        self.mllm_ckpt = config['mllm_ckpt']
        self.sample_frame = config['sample_frame']
        if 'llava' in self.mllm_ckpt:
            self.llm_type = 'llava'
        else:
            raise ValueError('Unsupported LLM type')
        self.img_encoder = stv_base_r50_s16_224_alpha(
            window_size=self.window_size, 
            pretrained=self.st_pretrained, 
            local_path=self.st_ckpt,
            in_chans=3,
            alpha=1.0
        )
        self.llava_forward_encoder = self.build_mm_encoder(model_path=self.mllm_ckpt, freezed_llm=freezed_llm)
        self.mm_indomain_proj = nn.Linear(4096, 1024)
        self.mm_multi_attn = CrossAttention(1024, 512)
        self.mm_multi_norm = nn.LayerNorm(1024)
        self.mm_patch_attn = CrossAttention(1024, 512)
        self.mm_patch_norm = nn.LayerNorm(1024)
        self.mm_visual_proj = nn.Linear(1024, 768)
        self.mm_textual_proj = nn.Linear(1024, 768)
        self.mm_final_attn = CrossAttention(768, 256)
        self.mm_final_norm = nn.LayerNorm(768)
        self.head = nn.Linear(768, 2)
        
        for p in self.llava_forward_encoder.parameters():
            p.requires_grad = False
        new_component_list = [
            self.mm_indomain_proj,
            self.mm_multi_attn,
            self.mm_patch_attn,
            self.mm_visual_proj,
            self.mm_textual_proj,
            self.mm_final_attn,
            self.head,
        ]
        for component in new_component_list:
            for m in component.modules():
                if isinstance(m, nn.Conv1d):
                    nn.init.kaiming_normal_(m.weight)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, std=0.01)
                    
    def build_mm_encoder(self, model_path, freezed_llm=True):
        assert self.llm_type
        if self.llm_type == 'llava':
            return LlavaForwardEncoder(model_path=model_path, freezed=freezed_llm)
        else:
            raise ValueError('Unsupported llm type')
        
    def forward(self, x_input, output_mm_states=False):
        B, L, C, H, W = x_input.shape
        st_feat = self.img_encoder(x_input)
        v_imgs = x_input[:, :self.sample_frame, :, :, :].permute(0, 1, 3, 4, 2)
        mm_visual = []
        mm_text = []
        for v in v_imgs:
            f_imgs = []
            for f in v:
                f_imgs.append(Image.fromarray(np.uint8(np.clip(f.cpu().numpy() * 255, 0, 255))).convert('RGB'))
            visual_embed, text_embed = self.llava_forward_encoder.forward_explicit(f_imgs, embed_text=True)
            mm_visual.append(visual_embed)
            mm_text.append(text_embed)
        mm_visual = torch.stack(mm_visual, dim=0)
        mm_text = torch.stack(mm_text, dim=0)
        # MultiFusion
        mm_indomain_text = self.mm_indomain_proj(mm_text)
        mm_cr = self.mm_multi_attn(mm_visual, mm_indomain_text)
        mm_cr = self.mm_multi_norm(mm_cr)
        
        mm_cr_cls, mm_cr_patch = mm_cr[:, 0, :].unsqueeze(1), mm_cr[:, 1:, :]
        mm_cr = self.mm_patch_attn(mm_cr_cls, mm_cr_patch)
        mm_cr = self.mm_patch_norm(mm_cr)
        
        mm_visual = self.mm_visual_proj(mm_cr)
        mm_multi = self.mm_final_attn(st_feat, mm_visual)
        mm_multi = self.mm_final_norm(mm_multi)
        out = self.head(torch.mean(mm_multi, dim=1))
        if output_mm_states:
            return out, mm_indomain_text, mm_cr_cls
        return out
    
    def assign_lr(self, module, lr, params_dict_list):
        params_dict_list.append({'params': module.parameters(), 'lr': lr})

    def assign_lr_dict_list(self, lr=1e-4):
        params_dict_list = []

        # backbone
        self.assign_lr(self.img_encoder, lr, params_dict_list)
        self.assign_lr(self.llava_forward_encoder, lr, params_dict_list)
        self.assign_lr(self.mm_indomain_proj, lr, params_dict_list)
        self.assign_lr(self.mm_multi_attn, lr, params_dict_list)
        self.assign_lr(self.mm_multi_norm, lr, params_dict_list)
        self.assign_lr(self.mm_patch_attn, lr, params_dict_list)
        self.assign_lr(self.mm_patch_norm, lr, params_dict_list)
        self.assign_lr(self.mm_visual_proj, lr, params_dict_list)
        self.assign_lr(self.mm_textual_proj, lr, params_dict_list)
        self.assign_lr(self.mm_final_attn, lr, params_dict_list)
        self.assign_lr(self.mm_final_norm, lr, params_dict_list)
        self.assign_lr(self.head, lr, params_dict_list)
        print('Finish assigning lr for model modules')
        return params_dict_list