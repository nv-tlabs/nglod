This folder contains the code for sparse training with Kaolin v0.9.1+.

To run (from `sdf-net`):

```
python app/spc/main_spc.py \
    --mesh-path data/armadillo.obj --normalize-mesh \
    --epoch 250 --base-lod 7 \
    --num-samples 5000000 --samples-per-voxel 32 \
    --exp-name armadillo-spc
```



