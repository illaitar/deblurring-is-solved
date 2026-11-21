# Evaluation protocol

This is the `protocol` preset of every method (option names in OPTIONS.md). Each method also has an `authors` preset
that reproduces its reported numbers; both are run and reported.

- **Weights**: the authors' GoPro-trained checkpoint on every test set (one set of weights, no fine-tuning). Released
  weights of other training sets are extra rows, marked.
- **Input**: whole frames, batch 1. GoPro uses `blur` (methods trained on `blur_gamma` are also run on it as an extra
  row). No tiling, no TLC / test-time local converter, no self-ensemble, no multi-pass averaging, unless the authors'
  own reported number needs it: then a second row with the authors' setting (`_tiled`, `_TLC`).
- **Padding**: replicate, only up to the network's stride, cropped back.
- **Precision**: fp16 autocast if its PSNR on the first 20 GoPro frames is within 0.05 dB of fp32, else fp32.
- **Video**: whole sequences, the authors' clip or window scheme; edge frames as the authors handle them. Frames a
  method does not emit are listed, and the means are reported both over emitted frames and over the common frame set.
- **Output**: 8-bit RGB (round to nearest), stored as FFV1 with a CRC per frame.
- **Each adapter is accepted when**: strict weight load, GoPro PSNR within 0.1 dB of the authors' reported value
  under their protocol (or the difference explained), and outputs stored for all four sets.
