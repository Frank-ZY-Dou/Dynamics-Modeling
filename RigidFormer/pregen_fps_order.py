"""Pre-generate order-independent full FPS orderings (obj_i_fpsorder_cf.npy) for every object of a dataset."""
import sys, glob, os, numpy as np, time
from pathlib import Path
repo=Path(sys.argv[1]); root=Path(sys.argv[2]); sys.path.insert(0,str(repo))
from src.data.multi_dataloader import MultiObjectDynamics
FPS=MultiObjectDynamics._farthest_point_sample_np
n_new=n_old=0; t0=time.time()
for s in sorted(glob.glob(str(root/'movi_*'))):
    f0=Path(s)/'frame_00000'; ad=Path(s)/'anchors'; ad.mkdir(exist_ok=True)
    for f in sorted(f0.glob('obj_*_pos.npy')):
        i=f.name.split('_')[1]; out=ad/f'obj_{i}_fpsorder_cf.npy'
        if out.exists(): n_old+=1; continue
        V=np.load(f).astype(np.float32); order=FPS(V,V.shape[0],rule='centroid_farthest')
        assert len(set(order.tolist()))==V.shape[0]
        tmp=out.with_name(out.stem+f'.{os.getpid()}.tmp.npy'); np.save(tmp,order.astype(np.int32)); os.replace(tmp,out); n_new+=1
print(f'[pregen-order] {root}: new {n_new}, existing {n_old}, {time.time()-t0:.0f}s')
