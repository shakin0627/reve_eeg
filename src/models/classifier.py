"""Downstream classifier heads and wrappers built on top of REVE encoder outputs."""

import torch
from einops.layers.torch import Rearrange
from torch import nn

from models.backbone import RMSNorm
from models.encoder import REVE
from utils.initialization import ConfigInit, init_cls
from models.norm import MongeNormLayer


H_DIM_MAP = {
    "cbramod": 200,
    "biot": 256,
    "labram": 1000,
}


class ReveClassifier(nn.Module):
    def __init__(
        self,
        encoder: REVE,
        n_classes,
        dropout,
        pooling="last",
        use_monge_norm=True,
        num_domains=None,
        monge_momentum=0.05,
        monge_recompute_every=50,
        monge_num_train_samples=None,
        **kwargs,
    ):
        super().__init__()
        assert pooling in ["last", "last_avg", "all", "no"], f"Pooling {pooling} not supported"
        self.encoder = encoder
        self.cls_query_token = nn.Parameter(torch.randn(1, 1, self.encoder.embed_dim))

        self.dropout = nn.Dropout(dropout)

        if pooling == "no":
            out_shape = kwargs.get("out_shape", self.encoder.embed_dim)
            assert isinstance(out_shape, int), "out_shape must be an integer"

            self.linear_head = nn.Sequential(
                Rearrange("b n d -> b (n d)"),
                RMSNorm(out_shape),
                self.dropout,
                nn.Linear(out_shape, n_classes),
            )
        else:
            self.linear_head = torch.nn.Sequential(
                RMSNorm(self.encoder.embed_dim),
                self.dropout,
                torch.nn.Linear(self.encoder.embed_dim, n_classes),
            )

        self.pooling = pooling

        # --- Monge normalization setup ---
        self.use_monge_norm = use_monge_norm
        if self.use_monge_norm:
            assert pooling in ["last", "all"], (
                f"use_monge_norm=True not supported for pooling={pooling}; "
                "MongeNormLayer is only wired into the 'last'/'all' branches."
            )
            assert num_domains is not None, "num_domains required when use_monge_norm=True"
            self.monge_norm = MongeNormLayer(
                feature_dim=self.encoder.embed_dim,
                num_domains=num_domains,
                momentum=monge_momentum,
                recompute_every=monge_recompute_every,
                num_train_samples=monge_num_train_samples,
            )


    def init_weights(self, config_megatron: ConfigInit):
        init_cls(self, config_megatron)
        print("Classifier weights initialized")

    def _apply_monge_norm(self, context, domain_ids=None, ot_weights=None, lam=None):
        if not self.use_monge_norm:
            return context
        if self.training:
            assert domain_ids is not None, "domain_ids required in train mode"
            return self.monge_norm(context, mode='train', domain_ids=domain_ids)
        else:
            assert ot_weights is not None and lam is not None, \
                "ot_weights and lam required in test mode"
            return self.monge_norm(context, mode='test', ot_weights=ot_weights, lam=lam)

    def forward(self, x, pos, return_attn=False, domain_ids=None, ot_weights=None, 
                lam=None, return_features=False, skip_monge_norm=False):
        if self.pooling == "last_avg":
            x = self.encoder(x, pos, False)
            x = x.mean(dim=1)
            return self.linear_head(x)
        elif self.pooling == "last":
            x = self.encoder(x, pos, False)
        elif self.pooling == "all":  # concatenate all intermediate layers
            x = torch.cat(self.encoder(x, pos, True), dim=1)
        elif self.pooling == "no":
            x = self.encoder(x, pos, False)
            b = x.shape[0]
            query_output = self.cls_query_token.expand(b, -1, -1)
            attention_scores = torch.matmul(query_output, x.transpose(-1, -2)) / (self.encoder.embed_dim**0.5)
            attention_weights = torch.softmax(attention_scores, dim=-1)  # (B, 1, L)
            context = torch.matmul(attention_weights, x)
            x = torch.cat([context, x], dim=-2)

            return self.linear_head(x)

        b = x.shape[0]
        query_output = self.cls_query_token.expand(b, -1, -1)
        attention_scores = torch.matmul(query_output, x.transpose(-1, -2)) / (self.encoder.embed_dim**0.5)
        attention_weights = torch.softmax(attention_scores, dim=-1)  # (B, 1, L)
        context = torch.matmul(attention_weights, x).squeeze(1)
        raw_context = context
        if skip_monge_norm:
            context = raw_context
        else:
            context = self._apply_monge_norm(context, domain_ids=domain_ids,
                                            ot_weights=ot_weights, lam=lam)
            
        out = self.linear_head(context)

        if return_attn and return_features:
            return out, attention_weights, raw_context
        if return_features:
            return out, raw_context
        if return_attn:
            return out, attention_weights
        
        return out

    def forward_attn(self, x, pos, domain_ids=None, ot_weights=None, lam=None):
        # returns prediction, query attention weights, and all intermediate attention weights
        x, attn = self.encoder.forward_attn(x, pos)
        b = x.shape[0]
        query_output = self.cls_query_token.expand(b, -1, -1)
        attention_scores = torch.matmul(query_output, x.transpose(-1, -2)) / (self.encoder.embed_dim**0.5)
        attention_weights = torch.softmax(attention_scores, dim=-1)
        context = torch.matmul(attention_weights, x).squeeze(1)
        # TODO: Monge Normalization
        context = self._apply_monge_norm(context, domain_ids=domain_ids,
                                          ot_weights=ot_weights, lam=lam)
        return self.linear_head(context), attention_weights, attn


def get_classifier(args, encoder):
    """
    Get the classifier model for downstream tasks
    The new weights are initialized using the init_cls function.
    """

    kwargs_ = args.task.classifier.kwargs if hasattr(args.task.classifier, "kwargs") else {}
    classifier = ReveClassifier(
        encoder=encoder,
        n_classes=args.task.classifier.n_classes,
        pooling=args.task.classifier.pooling,
        lp_dropout=args.task.linear_probing.dropout,
        ft_dropout=args.task.fine_tuning.dropout,
        **kwargs_,
    )

    classifier.init_weights(ConfigInit(**args.init))
    return classifier


class ClassifierWrapper(nn.Module):
    """
    A wrapper for the classifier to add a linear layer on top of the encoder.
    This is used for downstream tasks.
    """

    def __init__(self, model, args, h_dim):
        super().__init__()
        self.backbone = model
        # name of classification layer is consistent with the original model
        self.linear_head = nn.Linear(h_dim, args.task.classifier.n_classes)

    def forward(self, x, **kwargs):
        x = self.backbone(x, **kwargs)
        return self.linear_head(x)


def wrap_encoder(encoder, args):
    """
    Wrap the model with a classifier for downstream tasks.
    """
    if args.model_type == "reve":
        return get_classifier(args, encoder)
    else:
        h_dim = H_DIM_MAP[args.model_type]
        return ClassifierWrapper(encoder, args, h_dim)
