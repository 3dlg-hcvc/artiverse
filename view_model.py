import argparse
import csv
import datetime
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from tqdm import tqdm

import sys
sys.path.append("external")
import external.pygltftoolkit as pygltk


def _get_segmented_pids(model_dir, model_id):
    glb_path = f"{model_dir}/{model_id}.segmented.glb"
    if not os.path.exists(glb_path):
        raise FileNotFoundError(f"Segmented GLB not found: {glb_path}")
    gltf = pygltk.load(glb_path, annotated=True)
    return sorted(int(pid) for pid in np.unique(gltf.segmentation_map))


def render_model(model_dir, output_dir, args):
    model_id = model_dir.split("/")[-1]

    start_time = time.time()

    print(f"Processing {model_dir}")

    # Load articulations and precompute which pids have non-fixed/non-free motions
    articulations_json = f"{model_dir}/{model_id}.corrected.articulations.json"
    with open(articulations_json, "r") as f:
        arts_data = json.load(f)
    art_list = arts_data.get("articulations", [])

    has_non_fixed_by_pid = {}
    for art in art_list:
        pid = int(art.get("pid"))
        t = str(art.get("type", "fixed")).lower()
        if t not in ["fixed", "free"]:
            has_non_fixed_by_pid[pid] = True

    os.makedirs(output_dir, exist_ok=True)

    render_variants = [
        {"name": "videos", "render_mode": "highlight", "azimuth_offset_deg": 0.0},
        {"name": "videos_right", "render_mode": "highlight", "azimuth_offset_deg": 90.0},
        {"name": "videos_textured", "render_mode": "textured", "azimuth_offset_deg": 0.0},
        {"name": "videos_textured_right", "render_mode": "textured", "azimuth_offset_deg": 90.0},
    ]
    for variant in render_variants:
        os.makedirs(f"{output_dir}/{variant['name']}", exist_ok=True)

    segmented_pids = set(_get_segmented_pids(model_dir, model_id))
    articulated_non_fixed_pids = {pid for pid, has_non_fixed in has_non_fixed_by_pid.items() if has_non_fixed}

    # Render every segmented pid for verification.
    # Static pids will be rendered as 1-frame highlighted videos.
    render_pids = sorted(segmented_pids)
    missing_from_glb = sorted(articulated_non_fixed_pids.difference(segmented_pids))
    if missing_from_glb:
        print(f"[WARN] Articulated pids missing from segmented.glb for {model_id}: {missing_from_glb}")

    for pid in render_pids:
        is_non_fixed = bool(has_non_fixed_by_pid.get(pid, False))
        frames = 16 if is_non_fixed else 1
        animate_mode = "single" if is_non_fixed else "all"

        for variant in render_variants:
            variant_output_dir = f"{output_dir}/{variant['name']}"

            force_highlight_flag = ""
            if variant["render_mode"] == "highlight" and not is_non_fixed:
                force_highlight_flag = "--force_highlight"

            # HDRI strength for textured: controls how bright the studio environment appears.
            # Highlight mode keeps the user-supplied (low) ambient so magenta/gray colors stay vivid.
            ambient = 1.0 if variant["render_mode"] == "textured" else args.ambient_strength
            cmd = (
                f"blenderproc run utils/render_part_articulation.py "
                f"--model_input_path {model_dir} "
                f"--art_json_path {articulations_json} "
                f"--highlight_pid {pid} "
                f"--output_dir {variant_output_dir} "
                f"--frames {frames} "
                f"--fps {args.fps} "
                f"--animate_mode {animate_mode} "
                f"--render_mode {variant['render_mode']} "
                f"--azimuth_offset_deg {variant['azimuth_offset_deg']} "
                f"--ambient_strength {ambient} "
                f"--light_size_scale {args.light_size_scale} "
                f"{force_highlight_flag}"
            )

            print(f"Running: {cmd}")
            os.system(cmd)

    os.system(f"rm -rf _temp/{model_id}")

    end_time = time.time()




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--ambient_strength", type=float, default=0.08)
    parser.add_argument("--light_size_scale", type=float, default=1.8)
    args = parser.parse_args()

    render_model(args.model_path, args.output_dir, args)

if __name__ == "__main__":
    main()