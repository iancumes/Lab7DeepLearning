"""Audita resultados y checkpoints antes de publicar una entrega completa."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from gensim.models import KeyedVectors
from sklearn.metrics import precision_recall_fscore_support,confusion_matrix
from src.common import ART,CHECKPOINTS,load_json,save_json,sha256

def main(checkpoints=True):
    completion=load_json(ART/'completion.json')
    assert completion['complete'] and completion['sgns_epochs']==21
    assert completion['classifiers']==32 and completion['final_test_models']==20
    corpus=load_json(ART/'corpus.json'); assert corpus['selected_tokens']>=20_000_000
    assert sum(a['tokens'] for a in corpus['articles'])==corpus['selected_tokens']
    assert len({a['article_id'] for a in corpus['articles']})==corpus['selected_articles']
    assert corpus['article_boundaries'][-1]['tokens']==corpus['selected_tokens']
    assert corpus['selected_tokens']-corpus['articles'][-1]['tokens']<20_000_000
    configs=load_json(ART/'configurations.json'); metas=load_json(ART/'corpus_configurations.json')
    checked=[]
    candidates=[]
    for cfg,meta in zip(configs,metas):
        rows=load_json(ART/'runs'/f"{cfg['name']}.json")
        candidates.extend(rows)
        assert [r['epoch'] for r in rows]==[1,2,3]
        assert all(np.isfinite(r['loss']) and r['pairs']>0 for r in rows)
        assert all(r['corpus_tokens']==meta['corpus_tokens'] for r in rows)
        if checkpoints:
            assert sha256(ART/'ids'/cfg['name']/'corpus.txt')==meta['corpus_sha256']
            for epoch in [1,2,3]:
                state=torch.load(CHECKPOINTS/f"{cfg['name']}_e{epoch}.pt",map_location='cpu',weights_only=False)
                kv=KeyedVectors.load(str(CHECKPOINTS/f"{cfg['name']}_e{epoch}.kv"),mmap='r')
                assert state['corpus_sha256']==meta['corpus_sha256']
                assert kv.vector_size==cfg['dim'] and kv.index_to_key==meta['vocabulary']
                assert np.array_equal(kv.vectors,state['model']['input.weight'].numpy())
                assert torch.isfinite(state['model']['output.weight']).all()
                checked.append(f"{cfg['name']}_e{epoch}")
                del state,kv
    expected=max(candidates,key=lambda r:(r['wordsim']['spearman'] if r['wordsim']['spearman'] is not None else -2,r['analogies']['total']['accuracy'] or 0,-r['config']['dim'],-r['epoch']))
    observed=load_json(ART/'best_sgns.json')
    assert (expected['config']['name'],expected['epoch'])==(observed['config']['name'],observed['epoch'])
    if checkpoints:
        selected_vectors=KeyedVectors.load(str(CHECKPOINTS/f"{observed['config']['name']}_e{observed['epoch']}.kv"),mmap='r')
        published_vectors=KeyedVectors.load(str(CHECKPOINTS/'best_sgns.kv'),mmap='r')
        assert published_vectors.index_to_key==selected_vectors.index_to_key
        assert np.array_equal(published_vectors.vectors,selected_vectors.vectors)
        del selected_vectors,published_vectors
    gensim_logs=load_json(ART/'runs/gensim.json')
    assert len(gensim_logs)==3 and all(r['config']==observed['config'] for r in gensim_logs)
    if checkpoints:
        selected_meta=next(m for m in metas if m['config']['name']==observed['config']['name'])
        for epoch in [1,2,3]:
            kv=KeyedVectors.load(str(CHECKPOINTS/f'gensim_e{epoch}.kv'),mmap='r')
            assert set(kv.index_to_key)==set(selected_meta['vocabulary']) and kv.vector_size==observed['config']['dim']
            assert np.isfinite(kv.vectors).all()
            if kv.norms is not None: assert np.allclose(kv.norms,np.linalg.norm(kv.vectors,axis=1),rtol=1e-5,atol=1e-6)
            del kv
        final_vectors=KeyedVectors.load(str(CHECKPOINTS/'gensim.kv'),mmap='r')
        last_vectors=KeyedVectors.load(str(CHECKPOINTS/'gensim_e3.kv'),mmap='r')
        assert final_vectors.index_to_key==last_vectors.index_to_key
        assert np.array_equal(final_vectors.vectors,last_vectors.vectors)
        del final_vectors,last_vectors
    split=load_json(ART/'news_split.json'); train=set(split['train_indices']); val=set(split['validation_indices'])
    assert not train&val and len(train)==108000 and len(val)==12000 and split['official_test_size']==7600
    assert split['train_label_counts']==[27000]*4 and split['validation_label_counts']==[3000]*4
    subsets=[set(split['subsets'][str(f)]) for f in [.01,.1,.5,1.0]]
    assert all(subsets[i]<=subsets[i+1]<=train for i in range(3))
    selection=load_json(ART/'classification_selection.json'); tests=load_json(ART/'evaluations'/'classification_test.json')
    selected={r['key']:r for r in selection['selected']}
    assert len(selection['all_results'])==32 and len(selected)==len(tests)==20
    for row in tests:
        reference=selected[row['key']]
        assert row['checkpoint_sha256']==reference['checkpoint_sha256'] and row['test_evaluations']==1
        matrix=np.array(row['confusion_matrix']); assert matrix.sum()==7600 and matrix.shape==(4,4)
        pred=np.loadtxt(ART/'evaluations'/f"predictions_{row['key']}.csv",delimiter=',',skiprows=1,dtype=int)
        assert len(pred)==7600
        assert np.isclose((pred[:,1]==pred[:,2]).mean(),row['metrics']['accuracy'])
        assert np.isclose(matrix.trace()/matrix.sum(),row['metrics']['accuracy'])
        assert np.array_equal(matrix,confusion_matrix(pred[:,1],pred[:,2],labels=np.arange(4)))
        precision,recall,f1,_=precision_recall_fscore_support(pred[:,1],pred[:,2],average='macro',zero_division=0)
        for key,value in [('precision_macro',precision),('recall_macro',recall),('f1_macro',f1)]:
            assert np.isclose(value,row['metrics'][key])
        if checkpoints: assert sha256(ART/reference['checkpoint'])==reference['checkpoint_sha256']
    intrinsic=load_json(ART/'evaluations'/'intrinsic.json')
    assert len(load_json(ART/'shared_vocabulary.json'))==30000
    assert len({r['three_cos_add']['total']['evaluated'] for r in intrinsic.values()})==1
    for r in intrinsic.values():
        assert len(r['three_cos_add']['categories'])==len(r['three_cos_mul']['categories'])==14
        assert r['simlex']['total']==999 and r['wordsim']['total']==353
    save_json(ART/('verification.json' if checkpoints else 'verification_local.json'),dict(passed=True,checkpoints_verified=checked,sgns_epochs=21,
        classifiers=32,selected_test_models=20,test_examples=7600,shared_vocabulary=30000,
        corpus_minimum_passed=True,complete_articles=True,gensim_configuration_matches=True,
        gensim_vectors_verified=checkpoints,best_vectors_match_selection=checkpoints,
        partitions_disjoint=True,stratification_verified=True,selection_hashes_match=True,predictions_match_metrics=True))
    print('VERIFICACION_CORRECTA',len(checked),'checkpoints')

if __name__=='__main__': main('--no-checkpoints' not in sys.argv)
