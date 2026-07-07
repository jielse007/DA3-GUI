# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib

from depth_anything_3.specs import Prediction

# Lazy dispatch: submodules only loaded when their export_format is requested
_EXPORT_REGISTRY = {
    "glb": (".glb", "export_to_glb"),
    "mini_npz": (".npz", "export_to_mini_npz"),
    "npz": (".npz", "export_to_npz"),
    "feat_vis": (".feat_vis", "export_to_feat_vis"),
    "depth_vis": (".depth_vis", "export_to_depth_vis"),
    "gs_ply": (".gs", "export_to_gs_ply"),
    "gs_video": (".gs", "export_to_gs_video"),
    "colmap": (".colmap", "export_to_colmap"),
}


def _get_export_fn(format):
    mod_path, fn_name = _EXPORT_REGISTRY[format]
    mod = importlib.import_module(mod_path, package=__package__)
    return getattr(mod, fn_name)


def export(prediction, export_format, export_dir, **kwargs):
    if "-" in export_format:
        for fmt in export_format.split("-"):
            export(prediction, fmt, export_dir, **kwargs)
        return
    fn = _get_export_fn(export_format)
    fn(prediction, export_dir, **kwargs.get(export_format, {}))


__all__ = ["export"]
