"""Proves a WeSpeaker PyTorch checkpoint is the network inside an ONNX speaker model (Task 1).

Usage: python -I training/check_checkpoint.py CHECKPOINT ONNX [ONNX ...]

Fine-tuning starts from the checkpoint, so it must be the same network the app ships, or the
before/after comparison mixes two changes. This compares weights rather than outputs, so it runs
no model code: the checkpoint is read with torch.load(weights_only=True) and the ONNX files with
onnx's protobuf reader.

The exporter folds each BatchNorm into the convolution before it, so every checkpoint conv is
folded the same way (eps 1e-5, PyTorch's default, which WeSpeaker's ResNet keeps) and matched to
the ONNX conv initializer it is closest to. Also checks that the ONNX files share one graph, and
reports the largest difference found anywhere.
"""
import sys

import numpy as np
import onnx
import torch
from onnx import numpy_helper

EPS = 1e-5
TOLERANCE = 1e-5  # relative to each tensor's largest magnitude


def folded_convs(state):
    """(name, weight, bias) per conv, with the BatchNorm that follows it folded in."""
    convs = []
    for key in state:
        if not key.endswith(".weight") or state[key].dim() != 4:
            continue
        stem = key[: -len(".weight")]
        # conv1 -> bn1, layerX.Y.convN -> layerX.Y.bnN, layerX.Y.shortcut.0 -> layerX.Y.shortcut.1
        bn = stem[:-1] + "1" if stem.endswith("shortcut.0") else stem.replace("conv", "bn")
        w = state[key].double().numpy()
        gamma, beta = state[f"{bn}.weight"].double().numpy(), state[f"{bn}.bias"].double().numpy()
        mean, var = state[f"{bn}.running_mean"].double().numpy(), state[f"{bn}.running_var"].double().numpy()
        scale = gamma / np.sqrt(var + EPS)
        convs.append((stem, w * scale[:, None, None, None], beta - mean * scale))
    return convs


def relative(a, b):
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-12))


def main():
    state = torch.load(sys.argv[1], map_location="cpu", weights_only=True)
    convs = folded_convs(state)
    print(f"checkpoint: {len(state)} tensors, {len(convs)} convolutions, "
          f"projection head {tuple(state['projection.weight'].shape)} (dropped at export)")

    graphs = []
    for path in sys.argv[2:]:
        model = onnx.load(path)
        inits = {i.name: numpy_helper.to_array(i).astype(np.float64) for i in model.graph.initializer}
        conv_inits = [(inits[n.input[1]], inits[n.input[2]]) for n in model.graph.node if n.op_type == "Conv"]
        graphs.append([(n.op_type, list(n.input), list(n.output)) for n in model.graph.node])

        worst, unmatched = 0.0, []
        remaining = list(conv_inits)
        for stem, w, b in convs:
            candidates = [i for i, (ow, _) in enumerate(remaining) if ow.shape == w.shape]
            if not candidates:
                unmatched.append(stem)
                continue
            best = min(candidates, key=lambda i: relative(remaining[i][0], w))
            ow, ob = remaining.pop(best)
            worst = max(worst, relative(ow, w), relative(ob, b))
        for name, ckpt in (("model.seg_1.weight", "seg_1.weight"), ("model.seg_1.bias", "seg_1.bias")):
            worst = max(worst, relative(inits[name], state[ckpt].double().numpy()))
        mean_vec = inits["mean_vec"]

        ok = not unmatched and not remaining and worst <= TOLERANCE
        print(f"{path}: {len(conv_inits)} convs; largest relative difference {worst:.2e}; "
              f"mean_vec {'all zero' if not mean_vec.any() else f'norm {np.linalg.norm(mean_vec):.3g}'}"
              + (f"; unmatched {unmatched} / {len(remaining)} left over" if unmatched or remaining else "")
              + ("  -> SAME WEIGHTS" if ok else "  -> DIFFERENT"))
        if not ok:
            sys.exit(1)

    if len(graphs) > 1:
        same = all(g == graphs[0] for g in graphs[1:])
        print("ONNX graphs: " + ("identical node for node" if same else "DIFFER"))
        if not same:
            sys.exit(1)


if __name__ == "__main__":
    main()
