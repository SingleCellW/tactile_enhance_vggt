#!/usr/bin/env python3
"""Render auditable camera-intervention illustrations from completed Stage 1.b data.

These are slides of frozen physical states, not policy rollouts or a camera
trajectory. Selection uses test split/contact/occlusion only, never model error.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def font(size):
    for name in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                 "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"):
        if Path(name).exists():
            return ImageFont.truetype(name, size)
    return ImageFont.load_default()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.data / "manifest.json").read_text())
    if not manifest.get("complete"):
        raise ValueError("Camera data is incomplete")
    arrays = {key: np.load(args.data / (key + ".npy"), mmap_mode="r") for key in
              ("rgb", "split", "episode", "physical_sample", "camera_variant", "camera_domain",
               "pressure", "occlusion", "known_cameras")}
    selected = np.flatnonzero((arrays["split"] == 2) & (arrays["camera_variant"] == 0))
    active = (arrays["pressure"][selected] > .0005).any(-1)
    occlusion = arrays["occlusion"][selected].mean(-1)
    choices = []
    for name, mask, quantile in (("no_contact", ~active, .5), ("contact_visible", active, .1),
                                 ("contact_occluded", active, .9)):
        ids = np.flatnonzero(mask)
        if not len(ids):
            continue
        desired = np.quantile(occlusion[ids], quantile)
        chosen = ids[np.argmin(np.abs(occlusion[ids] - desired))]
        choices.append((name, int(selected[chosen])))
    args.output.mkdir(parents=True, exist_ok=True)
    items = []
    for name, nominal in choices:
        physical = int(arrays["physical_sample"][nominal])
        ids = np.flatnonzero(arrays["physical_sample"] == physical)
        ids = ids[np.argsort(arrays["camera_variant"][ids])]
        frames, transforms = [], []
        for row in ids:
            view = int(arrays["camera_variant"][row])
            domain = int(arrays["camera_domain"][row])
            label = "ORIGINAL VIEW" if domain == 0 else f"{'NEAR' if domain == 1 else 'STRONG'} VIEW {view if domain == 1 else view - 3}"
            canvas = Image.new("RGB", (960, 640), "#f5f6f2")
            draw = ImageDraw.Draw(canvas)
            draw.text((32, 19), "STAGE 1.b  /  CAMERA INTERVENTION", fill="#176a5c", font=font(17))
            draw.text((32, 50), label, fill="#203b36", font=font(27))
            for camera in range(2):
                frame = Image.fromarray(np.array(arrays["rgb"][row, camera])).resize((432, 432), Image.Resampling.NEAREST)
                canvas.paste(frame, (32 + camera * 464, 100))
                draw.text((42 + camera * 464, 110), ["LEFT", "RIGHT"][camera], fill="white", stroke_width=1, stroke_fill="#203b36", font=font(15))
            camera_shift = np.linalg.norm(arrays["known_cameras"][row, :, :3, 3] - arrays["known_cameras"][nominal, :, :3, 3], axis=-1) * 1000
            draw.text((32, 548), f"Frozen state {physical} | episode {int(arrays['episode'][row])} | 44 touch readings unchanged", fill="#203b36", font=font(18))
            draw.text((32, 580), f"Camera-center displacement L/R: {camera_shift[0]:.1f} / {camera_shift[1]:.1f} mm", fill="#6b7d75", font=font(16))
            draw.text((32, 610), "Independent rendered views. No physical time advances; this is not a policy rollout.", fill="#6b7d75", font=font(14))
            frames.append(canvas)
            transforms.append({"row": int(row), "camera_variant": view, "domain": domain,
                               "center_displacement_mm": camera_shift.tolist()})
        output = args.output / (name + ".mp4")
        command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-vcodec", "rawvideo",
                   "-pix_fmt", "rgb24", "-s", "960x640", "-r", "12", "-i", "-", "-an", "-c:v", "libx264",
                   "-preset", "fast", "-crf", "23", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output)]
        process = subprocess.Popen(command, stdin=subprocess.PIPE)
        try:
            for frame in frames:
                for _ in range(18):
                    process.stdin.write(frame.tobytes())
            process.stdin.close()
            if process.wait() != 0:
                raise RuntimeError("Video encoding failed")
        finally:
            if process.poll() is None:
                process.terminate()
        frames[0].save(args.output / (name + ".jpg"), quality=90)
        # The visual comparison explicitly uses original, near 1 and strong 1.
        strip = Image.new("RGB", (960, 640 * 3), "white")
        for k, index in enumerate((0, 1, 4)):
            strip.paste(frames[index], (0, k * 640))
        strip.save(args.output / (name + "_comparison.jpg"), quality=88)
        items.append({"id": name, "physical_sample": physical, "episode": int(arrays["episode"][nominal]),
                      "active_sensors": int((arrays["pressure"][nominal] > .0005).sum()),
                      "nominal_occlusion": arrays["occlusion"][nominal].tolist(), "views": transforms,
                      "seconds": len(frames) * 1.5, "fps": 12, "frame_hold_seconds": 1.5,
                      "file": output.name, "poster": name + ".jpg", "comparison": name + "_comparison.jpg",
                      "sha256": digest(output), "bytes": output.stat().st_size})
    catalog = {"complete": True, "data_manifest_sha256": digest(args.data / "manifest.json"),
               "selection": "Test original-camera rows: no-contact median occlusion, contact 10th and 90th percentile occlusion; closest row, stable first tie. No prediction/result selection.",
               "scope": "Frozen-state camera variants, not policy tests or physical-time trajectories", "videos": items}
    (args.output / "manifest.json").write_text(json.dumps(catalog, indent=2, allow_nan=False) + "\n")
    print(json.dumps(catalog, indent=2))


if __name__ == "__main__":
    main()
