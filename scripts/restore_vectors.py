"""Reconstruye .kv por epoch desde los checkpoints completos recuperados."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import numpy as np
from gensim.models import KeyedVectors
from src.common import ART,CHECKPOINTS,load_json

metas={m['config']['name']:m for m in load_json(ART/'corpus_configurations.json')}
for checkpoint in sorted(CHECKPOINTS.glob('*_e*.pt')):
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    meta=metas[state['config']['name']]
    assert state['corpus_sha256']==meta['corpus_sha256']
    vectors=state['model']['input.weight'].numpy()
    assert np.isfinite(vectors).all() and len(vectors)==len(meta['vocabulary'])
    destination=checkpoint.with_suffix('.kv')
    matching=False
    if destination.exists():
        previous=KeyedVectors.load(str(destination),mmap='r')
        matching=previous.index_to_key==meta['vocabulary'] and np.array_equal(previous.vectors,vectors)
        del previous
    if not matching:
        kv=KeyedVectors(vector_size=vectors.shape[1]); kv.add_vectors(meta['vocabulary'],vectors)
        kv.save(str(destination),separately=[])
        print('RESTAURADO',destination.name,flush=True)
print('Los checkpoints incluyen ambas matrices; los .kv conservan la de entrada.')
