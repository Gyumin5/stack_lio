# stack_lio

STACK = Self-contained IEKF-LIO modules (S-A3 + T-D + T-E + NO_RING) for geo-degeneracy-robust LiDAR-Inertial Odometry without per-sequence retuning.

Companion code to the STACK paper (target venue: IEEE Sensors Journal).

## Modules

- **S-A3** (`include/anchor_pool.h`): Fisher block min-eigenvalue rank-collapse detection + measurement quarantine
- **T-D** (`include/intensity_voxel_sketch.h`): 3D voxel NN re-ranking by intensity z-score consistency (ring-independent)
- **T-E** (`include/photometric_voxel_layer.h`, `include/tangent_patch.h`): 3D voxel intensity-map photometric residual added to IEKF Jacobian (λ=1e-5 default)
- **NO_RING**: Image-plane photometric residual disabled (ring-index-free mode)

## Origin

Derived from [COIN-LIO](https://github.com/ethz-asl/COIN-LIO) (ETH Zurich, MIT license). Original COIN-LIO contains the image-plane photometric residual (ring-dependent). STACK replaces it with 3D voxel-based intensity modules to remove the ring dependency while preserving intensity-aware geo-degeneracy robustness.

Upstream pristine COIN-LIO at: `<parent-repo>/catkin_ws/src/COIN-LIO/` (separate submodule).

License: MIT (preserved from upstream). See [LICENSE](./LICENSE).

## Build

```bash
cd catkin_ws/src && git submodule add git@github.com:Gyumin5/stack_lio.git
cd .. && catkin build stack_lio
source devel/setup.bash
```

## Launch

```bash
roslaunch stack_lio mapping_enwide.launch          # ENWide bags
roslaunch stack_lio mapping_newer_college.launch   # Newer College Col1/2/3/4
```

Environment flags:
- `NCC_NO_RING=1` — image-plane photometric OFF (STACK default)
- `NCC_LAMBDA_TE=1e-5` — T-E voxel residual gain (STACK default)

## Citation

(IEEE Sensors Journal submission in preparation.)
