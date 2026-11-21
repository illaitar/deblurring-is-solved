# Weights format

One file per model variant, no pickles:

```
weights/<method>/<variant>.safetensors     state dict as the adapter loads it (strict=True), original dtype
weights/<method>/<variant>.json            card
```

The card:

```json
{
  "method": "Restormer", "variant": "GoPro", "paper": "...", "venue": "CVPR 2022",
  "code": {"repo": "https://github.com/...", "commit": "<sha>"},
  "source": {"url": "<official link>", "file": "motion_deblurring.pth", "md5": "<md5 of the original file>"},
  "trained_on": "GoPro", "license": "...",
  "arch": {"...": "constructor arguments of the adapter"},
  "conversion": {"from": "pth", "keys_renamed": false, "dropped": [], "tool": "tools/convert_weights.py", "date": "..."},
  "check": {"strict_load": true, "tensors_equal_to_original": true}
}
```

Rules:

- The safetensors file holds exactly the tensors the model uses. Optimizer state, EMA copies the authors do not use
  at test time, and training-only heads are dropped and listed in `conversion.dropped`.
- Key renames (DataParallel prefixes, renamed modules) are listed. Values are never changed.
- `tools/convert_weights.py --verify` reloads the safetensors file and checks every tensor against the original
  checkpoint, bit for bit.
- Weights of other frameworks (JAX, TF1) are converted only through a port that is checked numerically against the
  original graph (PSS-NSC and SIUN: max difference 1e-6), and the card says so.
- Large files live on the cluster and, when public, on Hugging Face; git holds the cards only.
- Licenses are kept per model. A model whose license forbids redistribution keeps only its card and download link.
