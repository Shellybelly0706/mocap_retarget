"""Reusable smplx reconstruction helpers."""

from __future__ import annotations

import numpy as np


def as_tensor(value):
    import torch

    return torch.tensor(np.asarray(value), dtype=torch.float32)


def reconstruct_vertices(data, model_path, frame_ids, chunk_size=32):
    import torch
    from human_body_prior.body_model.body_model import BodyModel

    model = BodyModel(
        bm_fname=str(model_path),
        num_betas=int(data["num_betas"]),
        model_type="smplx",
    )
    faces = model.f.detach().cpu().numpy().astype(np.int32)
    vertices = None

    with torch.no_grad():
        for start in range(0, len(frame_ids), chunk_size):
            ids = frame_ids[start : start + chunk_size]
            count = len(ids)
            body = model(
                root_orient=as_tensor(data["root_orient"][ids]),
                pose_body=as_tensor(data["pose_body"][ids]),
                pose_hand=as_tensor(data["pose_hand"][ids]),
                pose_jaw=as_tensor(data["pose_jaw"][ids]),
                pose_eye=as_tensor(data["pose_eye"][ids]),
                betas=as_tensor(np.repeat(data["betas"][None], count, axis=0)),
                trans=as_tensor(data["trans"][ids]),
            )
            chunk = body.v.detach().cpu().numpy()
            if vertices is None:
                vertices = np.empty(
                    (len(frame_ids), chunk.shape[1], 3), dtype=np.float32
                )
            vertices[start : start + count] = chunk

    return vertices, faces
