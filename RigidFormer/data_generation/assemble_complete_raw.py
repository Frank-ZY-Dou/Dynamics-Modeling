"""Assemble the complete raw scenes of a MOVi dataset: every scene with every simulated object.

replay_movi_scene.py --write_root writes, for each scene whose metadata lacks never-visible objects (and whose recorded
objects it reproduced exactly), a completed metadata.json under <replayed>/<seed>/. This script builds a new root next
to HopNet's raw scenes and never modifies them:

    <out>/<seed>   symlink to <raw>/<seed> for a scene that is already complete; for a completed scene a directory with
                   the completed metadata.json and a symlink to the rendered rgba.mp4 (the recovered objects were never
                   visible, so the video is the same)

Every scene of <out> must then be complete (check_scenes.py). Convert <out> with movi_to_sim_data.py.

    python data_generation/assemble_complete_raw.py --raw raw/MOVi-B --replayed replay/MOVi-B --out raw/MOVi-B-complete
"""
import argparse
import json
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from check_scenes import exported_instances, replay_num_objects  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", required=True, help="HopNet's raw scenes <raw>/<seed>/metadata.json, rgba.mp4")
    ap.add_argument("--replayed", required=True, help="completed metadata from replay_movi_scene.py --write_root")
    ap.add_argument("--out", required=True, help="new folder for the complete raw scenes")
    args = ap.parse_args()

    raw, replayed, out = (pathlib.Path(p).resolve() for p in (args.raw, args.replayed, args.out))
    if out.exists():
        raise FileExistsError(f"{out} exists: assemble into a new folder")
    if not replayed.is_dir():
        raise FileNotFoundError(f"{replayed}: no replay output (replay_movi_scene.py --write_root creates it)")
    seeds = sorted(int(d.name) for d in raw.iterdir() if d.is_dir() and d.name.isdigit())
    done = {int(d.name) for d in replayed.iterdir() if d.is_dir() and d.name.isdigit()}
    if not done <= set(seeds):
        raise ValueError(f"replayed scenes not in {raw}: {sorted(done - set(seeds))}")

    # every completed metadata must keep the recorded instances unchanged and now be complete
    for s in sorted(done):
        orig = json.loads((raw / str(s) / "metadata.json").read_text())
        new = json.loads((replayed / str(s) / "metadata.json").read_text())
        k = len(orig["instances"])
        if new["instances"][:k] != orig["instances"] or new["flags"] != orig["flags"] or new["camera"] != orig["camera"]:
            raise ValueError(f"scene {s}: the completed metadata changes recorded content")
        if len(new["instances"]) != replay_num_objects(new) or len(orig["instances"]) == replay_num_objects(orig):
            raise ValueError(f"scene {s}: not an incomplete scene made complete")

    out.mkdir(parents=True)
    for s in seeds:
        if s in done:
            d = out / str(s)
            d.mkdir()
            (d / "metadata.json").write_bytes((replayed / str(s) / "metadata.json").read_bytes())
            os.symlink(raw / str(s) / "rgba.mp4", d / "rgba.mp4")
        else:
            os.symlink(raw / str(s), out / str(s))
    incomplete = [s for s in seeds if len(exported_instances(json.loads((out / str(s) / "metadata.json").read_text())))
                  != replay_num_objects(json.loads((out / str(s) / "metadata.json").read_text()))]
    if incomplete:
        raise RuntimeError(f"{len(incomplete)} scenes of {out} are still incomplete (not replayed?): {incomplete[:10]}")
    print(f"{len(seeds)} scenes: {len(seeds) - len(done)} linked, {len(done)} completed -> {out}")


if __name__ == "__main__":
    main()
