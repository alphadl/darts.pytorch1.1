# darts.pytorch1.1 — Code Review & Fixes

Review and fixes for this repo (PyTorch 1.1) vs [quark0/darts](https://github.com/quark0/darts) (PyTorch 0.3.1) and the DARTS paper (ICLR 2019).

---

## 1. Context

- **Original**: PyTorch 0.3.1 — uses `Variable` and old in-place API (e.g. `tensor.sub_(scale, other)`).
- **This repo**: PyTorch 1.1 — `Variable` removed (use `nn.Parameter`), in-place API updated to 1.x (e.g. `sub_(other, alpha=scale)`).

---

## 2. Fixes Applied

### 2.1 PyTorch 1.x in-place API (critical for `--unrolled`)

0.3: `tensor.sub_(scale, other)` → 1.x: `tensor.sub_(other, alpha=scale)` (same for `add_`).

- **cnn/architect.py**: `theta.sub(..., alpha=eta)`, `g.data.sub_(ig.data, alpha=eta)`, `p.data.add_(v, alpha=R)`, `p.data.sub_(v, alpha=2*R)`.
- **rnn/architect.py**: Same pattern; Hessian block updated with `alpha=`.

### 2.2 Architect `device_ids`

Replaced hardcoded `[0,1]` with `device_ids = [int(i) for i in getattr(args, 'gpu', '0').split(',')]`.

### 2.3 Unrolled model device

- **model_search.py** `new()`: use `next(self.parameters()).device` and `.to(device)`.
- **architect.py** `_construct_model_from_theta`: return `model_new.to(next(self.model.parameters()).device)` (same for rnn).

### 2.4 model.py

- Added `self.drop_path_prob = 0.0` in `NetworkCIFAR.__init__`.
- Removed debug `print` in `Cell.__init__`.

---

## 3. Notes (no code change)

- **PRIMITIVES**: This repo has 9 ops (no `sep_conv_7x7`, `conv_7x1_1x7`). Eval still has them in OPS for AmoebaNet etc.
- **Multi-GPU**: CNN keeps `DataParallel(model)` (no unwrap); `utils.save` writes `model.module.state_dict()` when wrapped; genotype/drop_path use `model.module if hasattr(model, 'module') else model`. RNN still single-GPU.
- **Variable**: Already migrated to `nn.Parameter`; no change.

---

## 4. Quick Check

```bash
cd cnn && python train_search.py --gpu 0
cd cnn && python train_search.py --gpu 0 --unrolled
cd cnn && python train.py --auxiliary --cutout --arch DARTS
```

---

## 5. Files Touched

| File | Changes |
|------|---------|
| cnn/architect.py | in-place API, device_ids, unrolled device |
| rnn/architect.py | in-place API, device_ids, unrolled device |
| cnn/model_search.py | `new()` uses current device |
| cnn/model.py | drop_path_prob init, remove print |
| cnn/train_search.py, train.py, utils.py | multi-GPU (keep DataParallel, save inner state_dict) |
