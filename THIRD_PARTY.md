# Upstream attribution and dependency snapshots

This is an experimental derivative of [MASt3R-SLAM](https://github.com/rmurai0610/MASt3R-SLAM)
by Riku Murai, Eric Dexheimer and Andrew J. Davison. The original `LICENSE.md`
and `Dependencies.md` are retained. The project uses the upstream
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license; each
third-party dependency retains its own license. It is not an official upstream release.

The source was copied from an existing working checkout, without its Git history.
This repository vendors the dependency source trees (ordinary tracked files) so that
local compatibility changes are retained. It does not claim that these snapshots
exactly match a particular upstream commit. Nested `.gitmodules` files are historical
metadata, not a requirement to initialize submodules in this snapshot.

| Path | Upstream / license location |
| --- | --- |
| `mast3r_slam`, original `main.py` | [MASt3R-SLAM](https://github.com/rmurai0610/MASt3R-SLAM), root `LICENSE.md` |
| `thirdparty/mast3r` | [MASt3R](https://github.com/naver/mast3r), its `LICENSE` |
| `thirdparty/mast3r/dust3r` | [DUSt3R](https://github.com/naver/dust3r), its `LICENSE` |
| `thirdparty/mast3r/dust3r/croco` | [CroCo](https://github.com/naver/croco), its `LICENSE` |
| `thirdparty/mast3r/asmk` | [ASMK](https://github.com/jenicek/asmk), its `LICENSE` |
| `thirdparty/eigen` | [Eigen](https://gitlab.com/libeigen/eigen), `COPYING.README` and accompanying license files |
| `thirdparty/in3d` | Visualization helper source distributed with the original MASt3R-SLAM checkout |
| `thirdparty/in3d/thirdparty/pyimgui` | [pyimgui](https://github.com/pyimgui/pyimgui), its `LICENSE` |
| `thirdparty/in3d/thirdparty/pyimgui/imgui-cpp` | [Dear ImGui](https://github.com/ocornut/imgui), its `LICENSE.txt` |

CUDA backend portions derived from DROID-SLAM are documented in `Dependencies.md`.
Model checkpoint licenses are separate; weights and downloaded datasets are not
included. See the upstream MASt3R checkpoint notice before downloading/using weights.

The added plane constraints, measured/model depth fusion, temporal depth correction,
bounded keyframe caching, disk-map viewer, pose graph and RGB-D transport are
experimental modifications. They are not a claim to reproduce another paper's full
SLAM system or its reported accuracy.
