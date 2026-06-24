"""Neural detector that turns a real RGB-D frame into :class:`Detection`s.

In the digital twin we get detections for free from ``RobotIO.semantics()``, so
this module is the *hardware* path (and the path used in photoreal Isaac runs
where we want to train perception too). The interface is intentionally identical
to the privileged channel so the mapper is agnostic.

Dryness estimation is the gardener-specific signal: a small head regresses a
leaf-wilt / soil-moisture proxy from the plant crop. Swap ``_DRYNESS_MODEL`` for
your trained checkpoint."""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..common.types import CameraFrame, Detection, LidarScan, SceneSemantics


class PlantDetector:
    """Detector seam. The default implementation raises unless a model is wired,
    to make it obvious that on hardware you must provide one (the sim path uses
    privileged semantics instead)."""

    def __init__(self, model: Optional[object] = None, device: str = "cuda"):
        self._model = model
        self.device = device

    def detect(self, camera: CameraFrame, lidar: Optional[LidarScan], stamp: float) -> SceneSemantics:
        if self._model is None:
            raise NotImplementedError(
                "No detection model wired. In simulation, prefer RobotIO.semantics(); "
                "on hardware, construct PlantDetector(model=<your YOLO/Mask2Former+dryness head>)."
            )
        # Expected model contract (documented, not enforced here):
        #   boxes, classes, world_xyz, dryness = self._model(camera.rgb, camera.depth, lidar)
        dets: list[Detection] = []
        outputs = self._model(camera, lidar)  # type: ignore[operator]
        for o in outputs:
            dets.append(
                Detection(
                    kind=o["kind"],
                    id=int(o["id"]),
                    position=np.asarray(o["position"], dtype=float),
                    attributes=o.get("attributes", {}),
                )
            )
        return SceneSemantics(stamp=stamp, detections=tuple(dets))
