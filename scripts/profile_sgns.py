"""Prueba de estabilidad y rendimiento con pares reales; no cuenta como epoch."""
from pathlib import Path
import sys,time,argparse
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from src.common import initialize,ART,SEED,load_json,save_json
from src.embeddings import SGNS
from src.preprocessing import pair_batches

def main(batch=4096,steps=2500):
    device=initialize(); rng=np.random.default_rng(SEED+1)
    folder=ART/'ids'/'base100'; meta=load_json(folder/'meta.json')
    ids=np.load(folder/'ids.npy',mmap_mode='r'); seg=np.load(folder/'segments.npy',mmap_mode='r')
    keep=np.load(folder/'keep.npy'); mask=rng.random(len(ids),dtype=np.float32)<keep[ids]
    ids,seg=np.asarray(ids[mask]),np.asarray(seg[mask])
    count=np.load(folder/'counts.npy'); weights=count**.75; cdf=np.cumsum(weights/weights.sum())
    model=SGNS(len(count),100).to(device); optimizer=torch.optim.SGD(model.parameters(),lr=.025)
    started=time.perf_counter(); total,pairs,maximum=0.,0,0.
    for step,(center,context) in enumerate(pair_batches(ids,seg,5,batch,rng)):
        neg=np.searchsorted(cdf,rng.random((len(center),5))); collision=neg==context[:,None]
        while collision.any():
            neg[collision]=np.searchsorted(cdf,rng.random(np.count_nonzero(collision)))
            collision=neg==context[:,None]
        optimizer.zero_grad(set_to_none=True)
        loss=model(torch.as_tensor(center.astype(np.int64),device=device),torch.as_tensor(context.astype(np.int64),device=device),torch.as_tensor(neg,device=device),reduction='sum')
        assert torch.isfinite(loss),f'Loss no finita en step {step}'
        loss.backward(); optimizer.step()
        value=float(loss.detach())/len(center); total+=value*len(center); pairs+=len(center); maximum=max(maximum,value)
        if step%500==0: print('PROFILE',batch,step,value,flush=True)
        if step+1==steps: break
    record=dict(batch_size=batch,steps=step+1,pairs=pairs,seconds=time.perf_counter()-started,
                mean_loss=total/pairs,last_loss=value,max_loss=maximum,finite=True,
                corpus_sha256=meta['corpus_sha256'],note='Prueba de rendimiento; excluida de selección y resultados de epochs.')
    save_json(ART/f'profile_sgns_{batch}.json',record)
    print(record,flush=True)
if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--batch',type=int,default=4096); parser.add_argument('--steps',type=int,default=2500)
    args=parser.parse_args(); main(args.batch,args.steps)
