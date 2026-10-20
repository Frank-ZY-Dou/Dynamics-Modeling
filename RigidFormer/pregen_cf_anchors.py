"""Pre-generate order-independent FPS anchor caches (obj_i_anchors_fps4_cf.npy) for every scene of a dataset."""
import sys, glob, os, numpy as np
from pathlib import Path
repo=Path(sys.argv[1]); root=Path(sys.argv[2]); K=int(sys.argv[3]) if len(sys.argv)>3 else 4
sys.path.insert(0,str(repo))
from src.data.multi_dataloader import MultiObjectDynamics
from src.data.scene_frames import load_frame
FPS=MultiObjectDynamics._farthest_point_sample_np
n_new=n_old=0
for s in sorted(glob.glob(str(root/'movi_*'))):
    ad=Path(s)/'anchors'; ad.mkdir(exist_ok=True)
    for i, V in enumerate(load_frame(Path(s), 0)):         # stored frame-0 vertices, per-file or packed layout
        out=ad/f'obj_{i}_anchors_fps{K}_cf.npy'
        if out.exists(): n_old+=1; continue
        idx=FPS(V.astype(np.float32),K,rule='centroid_farthest')
        tmp=out.with_name(out.stem+f'.{os.getpid()}.tmp.npy'); np.save(tmp,idx.astype(np.int32)); os.replace(tmp,out); n_new+=1
print(f'[pregen] {root}: new {n_new}, existing {n_old}')
