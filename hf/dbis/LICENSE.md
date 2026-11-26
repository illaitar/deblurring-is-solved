# Terms of the weight files

The files are converted from checkpoints released by the methods' authors. Each keeps the terms of its source; the
license texts are in `licenses/`.

| Files | Source checkpoint | Terms |
|---|---|---|
| `dbis-NAFNet-64.safetensors`, `dbisf-NAFNet-64.wpk` | `NAFNet-GoPro-width64.pth` of [megvii-research/NAFNet](https://github.com/megvii-research/NAFNet) (commit 2b4af71) | MIT, Copyright (c) 2022 megvii-model (`licenses/NAFNet-LICENSE.txt`) |
| `dbis-BSSTNet.safetensors`, `dbisf-BSSTNet.wpk` | `BSST_gopro.pth` of [huicongzhang/BSSTNet](https://github.com/huicongzhang/BSSTNet) (commit c35a43c) | MIT, as its README states ("This project is open sourced under MIT license"); the repository has no LICENSE file |
| the same two files | `raft-things.pth` of [princeton-vl/RAFT](https://github.com/princeton-vl/RAFT) | BSD 3-Clause, Copyright (c) 2020, princeton-vl (`licenses/RAFT-LICENSE.txt`) |
| `dbis-DSTNetPlus-L.safetensors`, `dbisf-DSTNetPlus-L.wpk` | `DSTNetPlus_L_gopro.pth` of [sunny2109/DSTNet-plus](https://github.com/sunny2109/DSTNet-plus) (release v0.1.0) | Apache License 2.0 (`licenses/DSTNet-plus-LICENSE.txt`) |
| `dbis-Turtle.safetensors`, `dbisf-Turtle.wpk` | `GoPro_Deblur.pth` of [Ascend-Research/Turtle](https://github.com/Ascend-Research/Turtle) (commit 7094f42) | MIT, Copyright (c) 2024 Huawei Technologies Co., Ltd. (`licenses/Turtle-LICENSE.txt`) |

Changes made to the source checkpoints (also stated for the Apache License 2.0, section 4): tensors renamed to the names
of our code, tensors folded into the next layers where that is exact up to float rounding, fixed tensors that the code
rebuilds dropped, and in the dbisf files the weights rounded per output channel and entropy coded. The conversion is
`python -m dbis.convert` of https://github.com/illaitar/deblurring-is-solved; it checks the md5 of every source
checkpoint (listed in each file's JSON card).
