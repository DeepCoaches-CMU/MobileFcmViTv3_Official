"""
P0.3 — trace the implemented MobileFCMViTv3 architecture from code and from
the headline checkpoint, in a clean checkout.

What it establishes (all written to one JSON file):
  1. The model instantiates from the headline run's config through the same
     build_model() the training script uses.
  2. The headline checkpoints load with strict=True: every tensor name and
     shape in the trained weights matches the code (both raw and EMA weights).
  3. Input contract: a (B,3,H,W) tensor works; a 4-channel tensor is rejected.
  4. Tensor shape at every stage of the forward pass (forward hooks).
  5. Parameter count per module and in total.
  6. Compute: thop (what the repo reports; counts MACs, misses functional
     matmuls) vs torch.utils.flop_counter (counts every matmul/conv, in FLOPs).
  7. Actual cluster behaviour of the trained weights on real validation
     images: membership spread and pairwise cosine of the learned centers.

Usage (from a clean checkout root, with the project venv):
  python audit/P0.3/scripts/trace_architecture.py \
      --config tuning/tune_05_augmix.yaml \
      --checkpoint <repo>/tuning/tune_05_augmix/checkpoints/model_best.pth.tar \
      --images <repo>/datasets/BUSI_split/validation --n_images 24 \
      --out audit/P0.3/architecture_trace.json
"""
import argparse
import copy
import hashlib
import json
import os
import random
import subprocess
import sys

import torch
import torch.nn.functional as F
import yaml

sys.path.insert(0, os.getcwd())
from mobilefcmvitv3.scripts.train import build_model  # noqa: E402  same factory as training
from mobilefcmvitv3.utils.augmentation import build_val_transform  # noqa: E402
from mobilefcmvitv3.utils.dataset import BUSIDataset  # noqa: E402


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def shape_of(x):
    if isinstance(x, torch.Tensor):
        return list(x.shape)
    if isinstance(x, (tuple, list)):
        return [shape_of(t) for t in x]
    return type(x).__name__


def trace_shapes(model, x):
    """Output shape of every top-level module and of the FCM sub-path."""
    names = ['conv_1', 'layer_1', 'layer_2', 'layer_3', 'layer_4', 'layer_5',
             'conv_1x1_exp', 'fcm_proj', 'clustering', 'fcm_fusion',
             'classifier.0', 'classifier.1', 'classifier.3']
    mods = dict(model.named_modules())
    rec, hooks = [], []
    for n in names:
        hooks.append(mods[n].register_forward_hook(
            lambda m, i, o, n=n: rec.append({'module': n, 'type': type(m).__name__,
                                             'in': shape_of(i), 'out': shape_of(o)})))
    with torch.no_grad():
        model(x)
    for h in hooks:
        h.remove()
    return rec


def param_breakdown(model):
    groups = {}
    for n, p in model.named_parameters():
        top = n.split('.')[0]
        groups[top] = groups.get(top, 0) + p.numel()
    total = sum(groups.values())
    return {'per_module': groups, 'total': total, 'total_M': round(total / 1e6, 4),
            'fcm_path_total': sum(v for k, v in groups.items()
                                  if k in ('fcm_proj', 'clustering', 'fcm_fusion'))}


def compute_counts(model, img=224):
    out = {}
    x = torch.randn(1, 3, img, img)
    try:
        from thop import profile
        m = copy.deepcopy(model)
        macs, _ = profile(m, inputs=(x,), verbose=False)
        out['thop_reported_G'] = round(macs / 1e9, 4)
        out['thop_note'] = 'thop counts multiply-accumulates (MACs) of nn modules only; repo logs this as "FLOPs"'
    except Exception as e:  # pragma: no cover
        out['thop_error'] = repr(e)
    from torch.utils.flop_counter import FlopCounterMode
    fc = FlopCounterMode(display=False)
    with torch.no_grad(), fc:
        model(x)
    flops = fc.get_total_flops()
    out['flop_counter_FLOPs_G'] = round(flops / 1e9, 4)
    out['flop_counter_MACs_G'] = round(flops / 2e9, 4)
    out['flop_counter_note'] = 'torch.utils.flop_counter: all aten matmul/conv ops incl. attention, FLOPs = 2 x MACs'
    return out


def cluster_behaviour(model, images_dir, n_images, seed=0):
    ds = BUSIDataset(images_dir, transform=build_val_transform(img_size=224))
    rng = random.Random(seed)
    idx = sorted(rng.sample(range(len(ds)), min(n_images, len(ds))))
    x = torch.stack([ds[i][0] for i in idx])
    captured = {}
    h = model.clustering.register_forward_hook(lambda m, i, o: captured.update(U=o[0], e=o[2]))
    with torch.no_grad():
        model(x)
    h.remove()
    U = captured['U']                        # (B, K, h, w)
    K = U.shape[1]
    c = F.normalize(model.clustering.centers.detach(), dim=-1)
    return {
        'n_images': len(idx), 'images_dir': images_dir, 'sample_seed': seed,
        'membership_shape': list(U.shape),
        'membership_min': float(U.min()), 'membership_max': float(U.max()),
        'max_abs_dev_from_uniform': float((U - 1.0 / K).abs().max()),
        'mean_entropy_term': float(captured['e']),
        'uniform_entropy_term_ln(K)/K': float(torch.log(torch.tensor(float(K))) / K),
        'center_raw_norms': [float(v) for v in model.clustering.centers.detach().norm(dim=-1)],
        'center_pairwise_cosine': [[round(float(v), 6) for v in row] for row in (c @ c.T)],
        'normalize_feat': model.clustering.normalize,
        'membership_mode': model.clustering.membership_mode,
        'tau': model.clustering.tau,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--extra_checkpoints', nargs='*', default=[])
    ap.add_argument('--images', required=True)
    ap.add_argument('--n_images', type=int, default=24)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    torch.manual_seed(0)
    cfg = yaml.safe_load(open(args.config))
    model = build_model(cfg).eval()
    report = {
        'code_version': subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True,
                                       text=True).stdout.strip() or 'not a git checkout (git archive export)',
        'cwd': os.getcwd(), 'torch': torch.__version__,
        'config': args.config, 'config_model_kwargs': {
            k: cfg.get(k) for k in ('model', 'fcm_k', 'fcm_proj_dim', 'tau', 'fcm_m', 'membership',
                                    'normalize_feat', 'fusion_type', 'dropout', 'attn_dropout',
                                    'ffn_dropout', 'drop_path_rate', 'lambda_clust', 'lambda_entropy',
                                    'img_size', 'pretrained_weights')},
        'model_class': type(model).__name__,
    }

    # Strict load of every checkpoint, raw and EMA weights.
    loads = []
    for path in [args.checkpoint] + args.extra_checkpoints:
        ck = torch.load(path, map_location='cpu', weights_only=False)
        entry = {'path': path, 'sha256': sha256(path), 'epoch': ck.get('epoch'), 'top1': ck.get('top1')}
        for key in ('state_dict', 'ema_state_dict'):
            if ck.get(key) is None:
                entry[key] = 'absent'
                continue
            m = build_model(cfg)
            try:
                m.load_state_dict(ck[key], strict=True)
                entry[key] = f'strict load OK ({len(ck[key])} tensors)'
            except RuntimeError as e:
                entry[key] = 'STRICT LOAD FAILED: ' + str(e)[:500]
        entry['stem_conv_weight_shape'] = list(ck['state_dict']['conv_1.block.conv.weight'].shape)
        entry['head_tensors'] = {k: list(v.shape) for k, v in ck['state_dict'].items()
                                 if k.startswith(('classifier', 'fcm_fusion', 'fcm_proj', 'clustering'))}
        loads.append(entry)
    report['checkpoint_loads'] = loads

    ck = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model.load_state_dict(ck['state_dict'], strict=True)
    model.eval()

    x3 = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        logits, c_loss, e_loss = model(x3)
    report['input_contract'] = {'accepts': [2, 3, 224, 224], 'returns': {
        'logits': list(logits.shape), 'compactness_loss': list(c_loss.shape),
        'entropy_loss': list(e_loss.shape)}}
    try:
        with torch.no_grad():
            model(torch.randn(2, 4, 224, 224))
        report['input_contract']['4_channel_input'] = 'ACCEPTED (unexpected)'
    except RuntimeError as e:
        report['input_contract']['4_channel_input'] = 'rejected: ' + str(e).split('\n')[0][:200]

    report['forward_shapes'] = trace_shapes(model, torch.randn(1, 3, 224, 224))
    report['parameters'] = param_breakdown(model)
    report['compute_224'] = compute_counts(model)
    report['trained_cluster_behaviour'] = cluster_behaviour(model, args.images, args.n_images)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(report, f, indent=2)
    print(json.dumps({k: report[k] for k in ('code_version', 'checkpoint_loads', 'input_contract',
                                             'parameters', 'compute_224', 'trained_cluster_behaviour')},
                     indent=1))


if __name__ == '__main__':
    main()
