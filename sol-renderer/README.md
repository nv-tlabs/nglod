# Neural SDF Renderer

This is the real-time rendering code.

### Build Instructions

First, make sure you have `cub` installed (it comes pre-installed for newer version of CUDA).

Then, download libtorch from the PyTorch website and extract as `libtorch` into the `third-party` directory. Then:

```
mkdir build
cd build
cmake ../
make -j8
```

To run,

```
./sdfRenderer {path_to_model.npz}
```

