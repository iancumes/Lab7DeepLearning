"""Pruebas de significado: objetivo SGNS, fronteras y aritmética vectorial."""
import numpy as np
import torch
from torch.nn import functional as F
from gensim.models import KeyedVectors
from src.preprocessing import tokenize,pair_count,pair_batches
from src.embeddings import SGNS,AnalogyEngine,read_analogies,read_similarity
from src.classification import NewsClassifier,encode,collate

def test_tokenizer_preserves_glove_compatible_information():
    assert tokenize("King's Café @-@ 3.14!")==["king's","café","-","3.14","!"]
    assert tokenize("= An Article =")==["an","article"]
    assert tokenize("<unk> <pad>")==[]

def test_pairs_do_not_cross_sentences_and_count_is_exact():
    ids=np.array([0,1,2,3,4],dtype=np.int32)
    segments=np.array([0,0,0,1,1],dtype=np.int32)
    batches=list(pair_batches(ids,segments,2,3,np.random.default_rng(23236),block_size=2))
    pairs=[(int(c),int(o)) for cs,os in batches for c,o in zip(cs,os)]
    expected={(0,1),(1,0),(1,2),(2,1),(0,2),(2,0),(3,4),(4,3)}
    assert set(pairs)==expected
    assert len(pairs)==pair_count(segments,2)==8

def test_sgns_matches_binary_cross_entropy_and_has_gradients():
    model=SGNS(5,3)
    with torch.no_grad():
        model.input.weight.copy_(torch.arange(15).reshape(5,3)/20)
        model.output.weight.copy_(torch.arange(15,30).reshape(5,3)/40)
    center=torch.tensor([0,1]); context=torch.tensor([2,3]); neg=torch.tensor([[3,4],[0,4]])
    positive=(model.input(center)*model.output(context)).sum(dim=1)
    negative=(model.input(center)[:,None,:]*model.output(neg)).sum(dim=2)
    manual=(F.binary_cross_entropy_with_logits(positive,torch.ones_like(positive),reduction="none")+
            F.binary_cross_entropy_with_logits(negative,torch.zeros_like(negative),reduction="none").sum(dim=1)).mean()
    loss=model(center,context,neg)
    assert torch.allclose(loss,manual)
    loss.backward()
    assert model.input.weight.grad.coalesce().values().abs().sum()>0
    assert model.output.weight.grad.coalesce().values().abs().sum()>0

def test_analogia_agrees_with_gensim_and_excludes_queries():
    rng=np.random.default_rng(42)
    kv=KeyedVectors(vector_size=10)
    kv.add_vectors([f"word{i}" for i in range(20)],rng.normal(size=(20,10)).astype(np.float32))
    engine=AnalogyEngine(kv)
    result=engine.analogia("word0","word1","word2",5)
    reference=kv.most_similar(positive=["word1","word2"],negative=["word0"],topn=5)
    assert [w for w,_ in result]==[w for w,_ in reference]
    assert np.allclose([s for _,s in result],[s for _,s in reference],atol=1e-5)
    assert not {"word0","word1","word2"}&set(w for w,_ in result)

def test_benchmark_files_have_expected_coverage_units():
    assert len(read_analogies())==19544
    assert len({q["category"] for q in read_analogies()})==14
    assert len(read_similarity())==353
    assert len(read_similarity("simlex999.txt"))==999

def test_embeddingbag_offsets_and_empty_document():
    encoded=encode([["a","b"],[],["a"]],["<pad>","<unk>","a","b"])
    ids,offsets=collate(encoded,[0,1,2],"cpu")
    assert offsets.tolist()==[0,2,3]
    model=NewsClassifier(torch.tensor([[0.,0.],[0.,0.],[1.,0.],[0.,1.]]),True)
    assert torch.allclose(model.embedding(ids,offsets),torch.tensor([[0.5,0.5],[0.,0.],[1.,0.]]))
    assert model(ids,offsets).shape==(3,4)
    assert not model.embedding.weight.requires_grad
