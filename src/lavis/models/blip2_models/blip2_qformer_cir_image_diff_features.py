"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import logging

import torch
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from torch.nn import functional as F

from lavis.common.registry import registry
from lavis.models.blip2_models.blip2 import (
    Blip2Base,
    disabled_train,
)
from utility import get_closs

@registry.register_model("blip2_cir_image_diff_features")
class Blip2QformerCirImageDiffFeatures(Blip2Base):
    """
    model with Q-former and ViT based on BLIP2.
    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("blip2_cir_image_diff_features", "pretrain")
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain": "configs/models/blip2/blip2_pretrain.yaml",
        "pretrain_vitL": "configs/models/blip2/blip2_pretrain_vitL.yaml",
        "coco": "configs/models/blip2/blip2_coco.yaml",
    }

    def __init__(
        self,
        vit_model="eva_clip_g",
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        num_query_token=32,
        cross_attention_freq=2,
        embed_dim=256,
        max_txt_len=32,
    ):
        super().__init__()

        self.tokenizer = self.init_tokenizer()

        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )
        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            logging.info("freeze vision encoder")
        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features, cross_attention_freq
        )
        self.Qformer.resize_token_embeddings(len(self.tokenizer))
        state_dict = self.Qformer.state_dict()
        for name, param in self.Qformer.named_parameters():
            if "_query" in name:
                key_orig = name.replace("_query", "")
                param.data.copy_(state_dict[key_orig])

        self.vision_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)
        self.text_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)
        
        self.d2t_proj = nn.Linear(self.visual_encoder.embed_dim, self.visual_encoder.embed_dim)

        self.itm_head = nn.Linear(self.Qformer.config.hidden_size, 2)

        self.temp = nn.Parameter(0.07 * torch.ones([]))

        self.max_txt_len = max_txt_len
        self.prompt_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, self.Qformer.config.hidden_size)
        )
        self.prompt_tokens.data.normal_(mean=0.0, std=self.Qformer.config.initializer_range)
    
    @torch.no_grad()
    def vit_encode(self, image):
        return self.visual_encoder(image)
    
    # Image Encoder
    def encode_image(self, image_embeds, query_tokens=None, ln=True):
        """ Encode images.
        Args:
            image_embeds (Tensor): Image representations encoded by ViT.
            query_tokens (Tensor): The query tokens of Q-Former.
            ln (Tensor): whether to perform layer norm.
        Returns:
            Tensor: Image representation encoded by Qformer.
        """
        if ln:
            with self.maybe_autocast():
                image_embeds = self.ln_vision(image_embeds)
        if query_tokens is None:
            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image_embeds.device
        )
        image_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        return image_output.last_hidden_state
    
    # Fusion Encoder
    def encode_fusion(self, F_image, text_tokens, 
                      no_image=False, diff_embeds=None, clean_label=None, pn_loss=None):
        """Fuse image representations with texts, image representations with image difference representations or prompt tokens with texts.
        
        Args:
            F_image (Tensor): , Image representation, or prompt tokens
            text_tokens (Tensor): text_tokens
            no_image (bool, optional): no_image is True if F_image is prompt tokens, Defaults to False.
            diff_embeds (Tensor optional): image difference encodings.
            clean_label (Tensor optional): the pseudo-labels indicating the cleanliness of triplets.
            pn_loss (dict optional): the loss settings.
        """
        bs = text_tokens.input_ids.shape[0]
        image_atts = torch.ones(F_image.shape[:-1], dtype=torch.long).to(
            F_image.device
        )
        attention_mask = torch.cat([image_atts, text_tokens.attention_mask], dim=1)
        if diff_embeds is not None:
            diff_atts = torch.ones(diff_embeds.shape[:-1], dtype=torch.long).to(
                F_image.device
            )
            attention_mask = torch.cat([image_atts, diff_atts], dim=1)
        assert F_image.shape[:-1] == (bs, 32)
        fusion_output = self.Qformer.bert(
            text_tokens.input_ids,
            query_embeds=F_image,
            attention_mask=attention_mask,
            return_dict=True,
            no_img=no_image,
            image_diff=diff_embeds,
            clean_label=clean_label,
            pn_loss=pn_loss,
        )
        if diff_embeds is not None:
            fusion_output, lsa = fusion_output
        token_num = 0 if no_image else 32
        res = F.normalize(self.text_proj(fusion_output.last_hidden_state[:, token_num, :]), dim=-1)
        res = (res, lsa) if diff_embeds is not None else res
        return res
    
    
    @torch.no_grad()
    def per_loss(self, reference_embeds, target_embeds, captions):
        F_r = self.encode_image(reference_embeds)
        F_t = self.encode_image(target_embeds)
        sim_i2t = self.inference(F_r, F_t, captions)
        loss = - (sim_i2t / self.temp).log_softmax(1).diag()
        return loss, sim_i2t.diag()

    def robust_infoNCE(self, scores, labels, pn_loss):
        eps=1e-7
        self.temp.data = torch.clamp(self.temp.data, min=1e-2)
        i2t = (scores/ self.temp).softmax(1)
        i2t = torch.clamp(i2t, min=eps, max=1-eps)
        target=torch.arange(scores.shape[0]).to(scores.device)
        clean_mask = labels.to(bool)
        noise_mask = ~clean_mask
        ploss = get_closs(i2t[clean_mask], target[clean_mask], pn_loss['positive_loss'])
        nloss = get_closs(i2t[noise_mask], target[noise_mask], pn_loss['negative_loss'])
        trade_off = pn_loss['trade_off']
        return trade_off * ploss + (1 - trade_off) * nloss

    def forward(self, samples, labels, pn_loss, warmup):
        image_embeds = samples["image"]
        target_embeds = samples["target"]
        text = samples["text_input"]
        image_embeds = self.ln_vision(image_embeds) # avoid repeated computation of laynorm in image encode
        text_tokens = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(image_embeds.device)
        text_tokens = text_tokens.to(image_embeds.device)
        # attention_mask = torch.cat([query_atts, text_tokens.attention_mask], dim=1)
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        F_reference = self.encode_image(image_embeds, query_tokens, ln=False)
        target_embeds = self.ln_vision(target_embeds)
        F_target = self.encode_image(target_embeds, query_tokens, ln=False)
        z_target = F.normalize(self.vision_proj(F_target), dim=-1)
        loss_dict = {}
        ###================ query encoding ================================###
        if warmup != 'proj':
            # fusion encode
            z_rm = self.encode_fusion(F_reference, text_tokens)
            sim_r2t = torch.matmul(
            z_rm.unsqueeze(1).unsqueeze(1), z_target.permute(0, 2, 1)
            ).squeeze()
            sim_r2t, _ = sim_r2t.max(-1)
            lrm = self.robust_infoNCE(sim_r2t, labels, pn_loss)
            loss_dict['lrm'] = lrm

        ###============== image diff encoding ===================###
        if warmup != 'qformer':
            image_diff_embds = self.d2t_proj(target_embeds - image_embeds)
            F_diff = self.encode_image(image_diff_embds, query_tokens, ln=False)
            # fussion encode
            z_rd, lsa = self.encode_fusion(F_reference, text_tokens, diff_embeds=F_diff, 
                                                 clean_label=labels, pn_loss=pn_loss)
            sim_d2t = torch.matmul(
                z_rd.unsqueeze(1).unsqueeze(1), z_target.permute(0, 2, 1)
            ).squeeze()
            sim_d2t, _ = sim_d2t.max(-1)
            local_labels = torch.ones(sim_d2t.shape[0]).to(image_embeds.device)
            lrd = self.robust_infoNCE(sim_d2t, local_labels, pn_loss)
            
        ###============== right prompt encoding ===================###
            prompt_tokens = self.prompt_tokens.expand(image_embeds.shape[0], -1, -1)
            z_pm = self.encode_fusion(prompt_tokens, text_tokens, no_image=True)
            # Prompt-modification query (reference-independent query) to target similarity
            sim_p2t = torch.matmul(
                z_pm.unsqueeze(1).unsqueeze(1), z_target.permute(0, 2, 1)
            ).squeeze()
            sim_p2t, _ = sim_p2t.max(-1)
            lpm = self.robust_infoNCE(sim_p2t, labels, pn_loss)
            loss_dict['lsa'] = lsa
            loss_dict['lpm'] = lpm
            loss_dict['lrd'] = lrd
        return loss_dict

    @torch.no_grad()
    def inference(self, F_reference, F_target, text):
        text_tokens = self.tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(F_reference.device)
        z_rm = self.encode_fusion(F_reference, text_tokens)
        z_target = F.normalize(self.vision_proj(F_target), dim=-1)
        sim_t2q = torch.matmul(
            z_rm.unsqueeze(1).unsqueeze(1), z_target.permute(0, 2, 1)
        ).squeeze()
        sim_i2t, _ = sim_t2q.max(-1)
        return sim_i2t

    @classmethod
    def from_config(cls, cfg):
        vit_model = cfg.get("vit_model", "eva_clip_g")
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")
        cross_attention_freq = cfg.get("cross_attention_freq", 2)

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)

        max_txt_len = cfg.get("max_txt_len", 32)

        model = cls(
            vit_model=vit_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            num_query_token=num_query_token,
            cross_attention_freq=cross_attention_freq,
            max_txt_len=max_txt_len,
        )
        model.load_checkpoint_from_config(cfg)

        return model
