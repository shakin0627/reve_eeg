"""Standalone downstream checkpoint evaluation script across subjects (Online Stream)"""

import os
import random
from types import SimpleNamespace

import hydra
import torch
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, f1_score
from tqdm import tqdm

from configs.resolver import register_resolvers
from downstream_tasks.dataloaders import get_streaming_eval_loader
from models.classifier import ReveClassifier
from models.encoder import REVE
from models.lam import LambdaController  
from models.tta import mahalanobis_cost, combine_lambda, softmax_transport_weights
from utils.model_utils import get_flattened_output_dim

register_resolvers()

# python src/dt_stream.py --config-name local_config.yaml --config-dir .

def _infer_num_train_samples(checkpoint_path: str) -> int:
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    key = "monge_norm.sorted_train_costs"
    if key not in state_dict:
        return None  
    return state_dict[key].numel()

class OnlineAdaptationEvaluator:
    """
    lam_mode:
      'off'            -- skip MongeNorm entirely (baseline 0)
      'static'         -- OT soft-assignment, lam=1 fixed (baseline 1)
      'lambda_b_only'  -- lam_A fixed at 1, only lam_B (transport entropy) varies
      'full'           -- lam_A (true EMA, per-subject) * lam_B, the actual online method
    """
    def __init__(self, model: ReveClassifier, lam_mode: str, alpha: float = 1.0,
                 eps_reg: float = 1e-5, strategy: str = "geometric",
                 lambda_a_momentum: float = 0.05):
        self.model = model
        self.lam_mode = lam_mode
        self.alpha = alpha
        self.eps_reg = eps_reg
        self.strategy = strategy

        self.use_monge = getattr(model, "use_monge_norm", True)
        self.controller = None
        if self.use_monge and lam_mode == "full":
            if not bool(model.monge_norm.cost_dist_ready):
                raise RuntimeError(
                    "lam_mode='full' requires finalize_train_cost_distribution() "
                    "to have been called after training (checkpoint должен already "
                    "contain sorted_train_costs)."
                )
            self.controller = LambdaController(model.monge_norm, momentum=lambda_a_momentum)

    def reset(self):
        """Call at each subject boundary (is_subject_start=True)."""
        if self.controller is not None:
            self.controller.reset()

    @torch.no_grad()
    def step(self, data: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        assert not self.model.training, "OnlineAdaptationEvaluator requires model.eval()"

        if self.lam_mode == "off" or not self.use_monge:
            return self.model(data, pos, skip_monge_norm=True)

        if not self.model.monge_norm.ready:
            return self.model(data, pos, skip_monge_norm=True)

        _, raw_context = self.model(data, pos, skip_monge_norm=True, return_features=True)

        ot_weights, lam_B = softmax_transport_weights(
            raw_context, self.model.monge_norm.running_mu, self.model.monge_norm.running_sigma,
            alpha=self.alpha, eps_reg=self.eps_reg,
        )

        if self.lam_mode == "static":
            lam = torch.ones_like(lam_B)
        elif self.lam_mode == "lambda_b_only":
            lam = combine_lambda(1.0, lam_B, strategy=self.strategy)
        elif self.lam_mode == "full":
            C = mahalanobis_cost(raw_context, self.model.monge_norm.running_mu,
                                  self.model.monge_norm.running_sigma, self.eps_reg)
            lam_A = self.controller.step(C)  # true EMA, persists across calls until reset()
            lam = combine_lambda(lam_A, lam_B, strategy=self.strategy)
        else:
            raise ValueError(f"unknown lam_mode: {self.lam_mode}")

        return self.model(data, pos, ot_weights=ot_weights, lam=lam)


def _run_stream(model, loader, lam_mode, device, alpha=1.0, strategy="geometric"):
    evaluator = OnlineAdaptationEvaluator(model, lam_mode=lam_mode, alpha=alpha, strategy=strategy)

    y_decisions, y_targets, y_subjects = [], [], []
    pbar = tqdm(loader, desc=f"Streaming eval (lam_mode={lam_mode})")
    for batch in pbar:
        if batch["is_subject_start"].item():
            evaluator.reset()

        data = batch["sample"].to(device, non_blocking=True)
        target = batch["label"].to(device, non_blocking=True)
        pos = batch["pos"].to(device, non_blocking=True)

        output = evaluator.step(data, pos)
        decision = torch.argmax(output, dim=1)

        y_decisions.append(decision.cpu())
        y_targets.append(target.cpu())
        y_subjects.append(batch["raw_subject_id"].item())

    gt = torch.cat(y_targets).numpy()
    pr = torch.cat(y_decisions).numpy()

    overall = {
        "Balanced Acc": balanced_accuracy_score(gt, pr),
        "Cohen Kappa": cohen_kappa_score(gt, pr),
        "F1 Score": f1_score(gt, pr, average="weighted"),
    }
    # TODO: per-subject balanced acc
    return overall


@hydra.main(version_base=None, config_name="config_dt", config_path="configs")
def main(args):
    device = args.trainer.device
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    backbone_args = SimpleNamespace(
        embed_dim=args.encoder.transformer.embed_dim,
        depth=args.encoder.transformer.depth,
        heads=args.encoder.transformer.heads,
        head_dim=args.encoder.transformer.head_dim,
        mlp_dim_ratio=args.encoder.transformer.mlp_dim_ratio,
        use_geglu=args.encoder.transformer.use_geglu,
    )
    encoder = REVE(
        args_backbone=backbone_args,
        freqs=args.encoder.freqs,
        patch_size=args.encoder.patch_size,
        overlap_size=args.encoder.patch_overlap,
        noise_ratio=args.encoder.noise_ratio,
    )

    n_chans = args.get("n_chans")
    n_timepoints = args.get("n_timepoints")
    if n_chans is None or n_timepoints is None:
        raise ValueError("n_chans and n_timepoints must be specified in the config")

    out_shape = None
    if args.task.classifier.pooling == "no":
        out_shape = get_flattened_output_dim(args, n_timepoints, n_chans)

    checkpoint_path = args.get("checkpoint_path", "model_best.pth")
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return

    inferred_num_train_samples = _infer_num_train_samples(checkpoint_path)

    model = ReveClassifier(
        encoder=encoder,
        n_classes=args.task.classifier.n_classes,
        dropout=args.get("dropout", 0.0),
        pooling=args.task.classifier.pooling,
        out_shape=out_shape,
        use_monge_norm=args.task.classifier.get("use_monge_norm", True),
        num_domains=args.task.classifier.get("num_domains", None),
        monge_momentum=args.task.classifier.get("monge_momentum", 0.05),
        monge_recompute_every=args.task.classifier.get("monge_recompute_every", 50),
        monge_num_train_samples=inferred_num_train_samples,
    )

    checkpoint_path = args.get("checkpoint_path", "model_best.pth")
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return

    print(f"Loading weights from {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)  # strict=True
    model.to(device)
    model.eval()

    lmdb_path = args.task.data_loader.dataset.path  
    lam_modes = ["off", "full"]  

    for split in ["val", "test"]:
        loader = get_streaming_eval_loader(lmdb_path, split)
        print(f"\n>>> Streaming evaluation on {split} split...")
        for lam_mode in lam_modes:
            metrics = _run_stream(model, loader, lam_mode=lam_mode, device=device)
            print(f"  [{lam_mode}] {metrics}")

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()