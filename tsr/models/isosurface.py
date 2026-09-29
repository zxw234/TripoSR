from typing import Callable, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

try:
    from torchmcubes import marching_cubes as _torch_mcubes

    def _marching_cubes(volume: torch.Tensor, isovalue: float):
        return _torch_mcubes(volume, isovalue)

except Exception:
    from skimage.measure import marching_cubes as _sk_mcubes

    def _marching_cubes(volume: torch.Tensor, isovalue: float):
        # skimage fallback (no CUDA compile needed)
        vol = volume.detach().cpu().numpy().astype(np.float32)
        verts, faces, _normals, _values = _sk_mcubes(vol, level=float(isovalue))
        v = torch.from_numpy(verts.astype(np.float32))
        f = torch.from_numpy(faces.astype(np.int64))
        return v, f


class IsosurfaceHelper(nn.Module):
    points_range: Tuple[float, float] = (0, 1)

    @property
    def grid_vertices(self) -> torch.FloatTensor:
        raise NotImplementedError


class MarchingCubeHelper(IsosurfaceHelper):
    def __init__(self, resolution: int) -> None:
        super().__init__()
        self.resolution = resolution
        self.mc_func: Callable = _marching_cubes
        self._grid_vertices: Optional[torch.FloatTensor] = None

    @property
    def grid_vertices(self) -> torch.FloatTensor:
        if self._grid_vertices is None:
            # keep the vertices on CPU so that we can support very large resolution
            x, y, z = (
                torch.linspace(*self.points_range, self.resolution),
                torch.linspace(*self.points_range, self.resolution),
                torch.linspace(*self.points_range, self.resolution),
            )
            x, y, z = torch.meshgrid(x, y, z, indexing="ij")
            verts = torch.cat(
                [x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)], dim=-1
            ).reshape(-1, 3)
            self._grid_vertices = verts
        return self._grid_vertices

    def forward(
        self,
        level: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, torch.LongTensor]:
        level = -level.view(self.resolution, self.resolution, self.resolution)
        try:
            v_pos, t_pos_idx = self.mc_func(level.detach(), 0.0)
        except AttributeError:
            print("torchmcubes was not compiled with CUDA support, use CPU version instead.")
            v_pos, t_pos_idx = self.mc_func(level.detach().cpu(), 0.0)
        # skimage returns XYZ already in array index order; torchmcubes needs axis swap
        if v_pos.shape[-1] == 3:
            # Keep consistent with upstream torchmcubes path when available
            try:
                from torchmcubes import marching_cubes as _  # noqa: F401
                v_pos = v_pos[..., [2, 1, 0]]
            except Exception:
                pass
        v_pos = v_pos / (self.resolution - 1.0)
        return v_pos.to(level.device), t_pos_idx.to(level.device)
