"""SGNS propio en PyTorch y evaluación intrínseca de vectores de palabras."""
from __future__ import annotations
import itertools
import math
import shutil
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from gensim.models import KeyedVectors, Word2Vec
from gensim.models.word2vec import LineSentence
from gensim.models.callbacks import CallbackAny2Vec
from gensim.test.utils import datapath
from scipy.stats import spearmanr
import gensim.downloader as api
from .common import ART, CHECKPOINTS, CONTROL, SEED, save_json, load_json, hardware
from .preprocessing import pair_batches, pair_count

def read_analogies():
    questions, category = [], None
    for line in Path(datapath("questions-words.txt")).read_text(encoding="utf-8").splitlines():
        if line.startswith(":"):
            category = line[1:].strip()
        elif line.strip():
            words = line.lower().split()
            questions.append(dict(category=category, words=words,
                                  kind="syntactic" if category.startswith("gram") else "semantic"))
    assert len(questions)==19544
    return questions

def read_similarity(name="wordsim353.tsv"):
    pairs = []
    for line in Path(datapath(name)).read_text(encoding="utf-8").splitlines():
        columns = line.split()
        if len(columns)<3 or line.startswith("#"):
            continue
        try:
            score=float(columns[2])
        except ValueError:
            continue
        pairs.append((columns[0].lower(), columns[1].lower(), score))
    expected = 999 if "simlex" in name else 353
    assert len(pairs)==expected, (name,len(pairs))
    return pairs

def similarity_score(kv, pairs):
    valid = [(a,b,s) for a,b,s in pairs if a in kv and b in kv]
    model = [float(kv.similarity(a,b)) for a,b,_ in valid]
    human = [s for _,_,s in valid]
    rho = spearmanr(model,human).statistic if len(valid)>1 else float("nan")
    return dict(spearman=float(rho) if math.isfinite(rho) else None,
                evaluated=len(valid), total=len(pairs), coverage=len(valid)/len(pairs))

class SGNS(nn.Module):
    """Dos matrices W y W'; pérdida estable con log-sigmoid.

    Los negativos llegan como IDs: la clase no depende de gensim.Word2Vec.
    Se usa suma para actualizar con SGD por contribuciones de pares; al
    reportar la pérdida se divide por la cantidad de pares observada.
    """
    def __init__(self, vocabulary_size, dimension):
        super().__init__()
        self.input = nn.Embedding(vocabulary_size, dimension, sparse=True)
        self.output = nn.Embedding(vocabulary_size, dimension, sparse=True)
        nn.init.uniform_(self.input.weight,-0.5/dimension,0.5/dimension)
        nn.init.zeros_(self.output.weight)

    def forward(self, center, context, negatives, reduction="mean"):
        v, u, n = self.input(center), self.output(context), self.output(negatives)
        positive = (v*u).sum(dim=1)
        negative = torch.einsum("bd,bkd->bk",v,n)
        losses = -F.logsigmoid(positive)-F.logsigmoid(-negative).sum(dim=1)
        return losses.sum() if reduction=="sum" else losses.mean()

class AnalogyEngine:
    """3CosAdd/3CosMul sobre un conjunto explícito de candidatos."""
    def __init__(self, kv, device="cpu", vocabulary=None):
        self.kv = kv
        self.words = vocabulary if vocabulary is not None else kv.index_to_key
        self.index = {w:i for i,w in enumerate(self.words)}
        self.device=torch.device(device)
        self.vectors = torch.as_tensor(np.asarray([kv[w] for w in self.words]), device=self.device)
        self.vectors = F.normalize(self.vectors,dim=1)

    def query_vector(self,a,b,c):
        query = sum(sign*self.kv.get_vector(w,norm=True) for sign,w in [(-1,a),(1,b),(1,c)])
        return torch.as_tensor(query,device=self.device)

    def analogia(self,a,b,c,k=5,exclude=True):
        """a es a b como c es a ?: normaliza cada vector de la consulta."""
        query = F.normalize(self.query_vector(a,b,c),dim=0)
        scores = self.vectors@query
        if exclude:
            for word in {a,b,c}:
                if word in self.index:
                    scores[self.index[word]]=-torch.inf
        values,indices=torch.topk(scores,min(k,len(self.words)))
        return [(self.words[i],float(s)) for i,s in zip(indices.cpu().tolist(),values.cpu().tolist())]

    def benchmark(self, questions, method="3CosAdd", batch_size=128):
        covered=[q for q in questions if all(w in self.index for w in q["words"])]
        correct={category:0 for category in dict.fromkeys(q["category"] for q in questions)}
        evaluated={category:0 for category in correct}
        for start in range(0,len(covered),batch_size):
            batch=covered[start:start+batch_size]
            ids=torch.tensor([[self.index[w] for w in q["words"]] for q in batch],device=self.device)
            a,b,c=(self.vectors[ids[:,j]] for j in range(3))
            if method=="3CosAdd":
                scores=F.normalize(b-a+c,dim=1)@self.vectors.T
            else:
                # Desplazamiento coseno a [0,1], definición de gensim.
                sa=(a@self.vectors.T+1)/2
                sb=(b@self.vectors.T+1)/2
                sc=(c@self.vectors.T+1)/2
                scores=sb*sc/(sa+1e-6)
            rows=torch.arange(len(batch),device=self.device)[:,None]
            scores[rows,ids[:,:3]]=-torch.inf
            hits=(scores.argmax(dim=1)==ids[:,3]).cpu().tolist()
            for q,hit in zip(batch,hits):
                evaluated[q["category"]]+=1
                correct[q["category"]]+=int(hit)
        categories=[]
        for category in correct:
            n=sum(q["category"]==category for q in questions)
            e=evaluated[category]
            categories.append(dict(category=category,total=n,evaluated=e,correct=correct[category],
                                   coverage=e/n,accuracy=correct[category]/e if e else None))
        overall={}
        for name in ["semantic","syntactic","total"]:
            rows=[r for r in categories if name=="total" or (r["category"].startswith("gram"))==(name=="syntactic")]
            e=sum(r["evaluated"] for r in rows)
            h=sum(r["correct"] for r in rows)
            n=sum(r["total"] for r in rows)
            overall[name]=dict(accuracy=h/e if e else None,evaluated=e,correct=h,total=n,coverage=e/n)
        return dict(method=method,candidate_vocabulary=len(self.words),categories=categories,**overall)

def epoch_evaluation(kv, selection_words, device):
    engine=AnalogyEngine(kv,device,selection_words)
    analogies=engine.benchmark(read_analogies())
    neighbors={word:kv.most_similar(word,topn=5) if word in kv else [] for word in CONTROL}
    fixed_path=ART/"selection_wordsim.json"
    pairs=load_json(fixed_path) if fixed_path.exists() else read_similarity()
    score=similarity_score(kv,pairs)
    score.update(total=353,coverage=score["evaluated"]/353,fixed_selection_pairs=True)
    return dict(analogies=analogies, wordsim=score,neighbors=neighbors)

def load_glove():
    cached=CHECKPOINTS/"glove.kv"
    if cached.exists():
        return KeyedVectors.load(str(cached),mmap="r")
    started=time.perf_counter()
    kv=api.load("glove-wiki-gigaword-100")
    kv.save(str(cached))
    save_json(ART/"glove_source.json",dict(model="glove-wiki-gigaword-100",tokens=6_000_000_000,
              vocabulary_size=len(kv),dimension=kv.vector_size,download_load_seconds=time.perf_counter()-started,
              source="https://nlp.stanford.edu/projects/glove/",
              published_runtime_note="El artículo reporta 85 min para coocurrencias (un hilo) y 14 min por iteración de 300d (32 cores), dual Intel Xeon E5-2658 2.1 GHz. No es un tiempo medido de glove.6B.100d."))
    return kv

def train_sgns(config, meta, selection_words, device):
    folder=ART/"ids"/config["name"]
    ids=np.load(folder/"ids.npy",mmap_mode="r")
    segments=np.load(folder/"segments.npy",mmap_mode="r")
    frequencies=np.load(folder/"counts.npy")
    keep=np.load(folder/"keep.npy")
    weights=np.power(frequencies,config["ns_exponent"])
    cdf=np.cumsum(weights/weights.sum())
    torch.manual_seed(config["seed"])
    model=SGNS(len(meta["vocabulary"]),config["dim"]).to(device)
    optimizer=torch.optim.SGD(model.parameters(),lr=config["initial_lr"])
    log_path=ART/"runs"/(config["name"]+".json")
    logs=load_json(log_path) if log_path.exists() else []
    if logs:
        last=logs[-1]["epoch"]
        checkpoint=torch.load(CHECKPOINTS/f'{config["name"]}_e{last}.pt',map_location=device,weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
    started=time.perf_counter()
    for epoch in range(len(logs)+1,config["epochs"]+1):
        epoch_started=time.perf_counter()
        rng=np.random.default_rng(config["seed"]+epoch)
        retained=rng.random(len(ids),dtype=np.float32)<keep[ids]
        sampled_ids,sampled_segments=np.asarray(ids[retained]),np.asarray(segments[retained])
        total_pairs=pair_count(sampled_segments,config["window"])
        dropped={word:dict(meta["subsampling"][word], observed_discard_fraction=
                 1-np.count_nonzero(sampled_ids==meta["vocabulary"].index(word))/meta["subsampling"][word]["occurrences"])
                 for word in meta["subsampling"]}
        print("START_SGNS",config["name"],epoch,"pairs",total_pairs,"retained_tokens",len(sampled_ids),flush=True)
        if device.type=="cuda":
            torch.cuda.reset_peak_memory_stats()
        loss_sum,pairs_seen=0.0,0
        for step,(centers,contexts) in enumerate(pair_batches(sampled_ids,sampled_segments,config["window"],config["batch_size"],rng)):
            # Rechazar el contexto positivo para evitar etiquetas contradictorias.
            negatives=np.searchsorted(cdf,rng.random((len(centers),config["negatives"])))
            collision=negatives==contexts[:,None]
            while collision.any():
                negatives[collision]=np.searchsorted(cdf,rng.random(np.count_nonzero(collision)))
                collision=negatives==contexts[:,None]
            progress=((epoch-1)+pairs_seen/max(total_pairs,1))/config["epochs"]
            optimizer.param_groups[0]["lr"]=config["initial_lr"]+(config["final_lr"]-config["initial_lr"])*progress
            center=torch.as_tensor(centers.astype(np.int64),device=device)
            context=torch.as_tensor(contexts.astype(np.int64),device=device)
            negative=torch.as_tensor(negatives.astype(np.int64),device=device)
            optimizer.zero_grad(set_to_none=True)
            loss=model(center,context,negative,reduction="sum")
            if not torch.isfinite(loss):
                raise RuntimeError(f"Pérdida no finita: {config['name']}, epoch {epoch}, step {step}")
            loss.backward()
            optimizer.step()
            loss_sum+=float(loss.detach())
            pairs_seen+=len(centers)
            if step%2000==0:
                print("SGNS",config["name"],epoch,"step",step,"pairs",pairs_seen,"loss",loss_sum/max(pairs_seen,1),flush=True)
        assert pairs_seen==total_pairs, (pairs_seen,total_pairs)
        if device.type=="cuda":
            torch.cuda.synchronize()
        training_seconds=time.perf_counter()-epoch_started
        peak=torch.cuda.max_memory_allocated() if device.type=="cuda" else None
        kv=KeyedVectors(vector_size=config["dim"])
        kv.add_vectors(meta["vocabulary"],model.input.weight.detach().cpu().numpy())
        kv.save(str(CHECKPOINTS/f'{config["name"]}_e{epoch}.kv'))
        torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),config=config,
                        corpus_sha256=meta["corpus_sha256"]),CHECKPOINTS/f'{config["name"]}_e{epoch}.pt')
        evaluation_started=time.perf_counter()
        evaluation=epoch_evaluation(kv,selection_words,device)
        record=dict(epoch=epoch,config=config,corpus_tokens=meta["corpus_tokens"],vocabulary_size=len(kv),
                    loss=loss_sum/max(pairs_seen,1),pairs=pairs_seen,retained_tokens=len(sampled_ids),
                    training_seconds=training_seconds,gpu_peak_bytes=peak,
                    evaluation_seconds=time.perf_counter()-evaluation_started,subsampling=dropped,**evaluation)
        logs.append(record)
        save_json(log_path,logs)
        print("END_SGNS",config["name"],epoch,"rho",record["wordsim"]["spearman"],"accuracy",record["analogies"]["total"]["accuracy"],flush=True)
    return logs

def select_best(configs):
    rows=[r for cfg in configs for r in load_json(ART/"runs"/(cfg["name"]+".json"))]
    def score(r):
        return (r["wordsim"]["spearman"] if r["wordsim"]["spearman"] is not None else -2,
                r["analogies"]["total"]["accuracy"] or 0,-r["config"]["dim"],-r["epoch"])
    best=max(rows,key=score)
    save_json(ART/"best_sgns.json",best)
    kv=KeyedVectors.load(str(CHECKPOINTS/f'{best["config"]["name"]}_e{best["epoch"]}.kv'))
    kv.save(str(CHECKPOINTS/"best_sgns.kv"),separately=[])
    return best,kv

def train_gensim(best,meta,selection_words,device):
    final=CHECKPOINTS/"gensim.kv"
    if final.exists() and (ART/"runs"/"gensim.json").exists():
        return KeyedVectors.load(str(final)),load_json(ART/"runs"/"gensim.json")
    cfg=best["config"]
    sentences=LineSentence(str(ART/"ids"/cfg["name"]/"corpus.txt"))
    model=Word2Vec(vector_size=cfg["dim"],window=cfg["window"],min_count=cfg["min_count"],
                  sample=cfg["sample"],sg=1,hs=0,negative=cfg["negatives"],ns_exponent=0.75,
                  alpha=cfg["initial_lr"],min_alpha=cfg["final_lr"],workers=4,
                  seed=cfg["seed"],epochs=cfg["epochs"],shrink_windows=False,compute_loss=True)
    model.build_vocab(sentences)
    assert set(model.wv.index_to_key)==set(meta["vocabulary"])
    class Recorder(CallbackAny2Vec):
        def __init__(self):
            self.logs=[]
            self.previous=0
        def on_epoch_begin(self,model):
            self.started=time.perf_counter()
        def on_epoch_end(self,model):
            training_seconds=time.perf_counter()-self.started
            cumulative=model.get_latest_training_loss()
            loss=cumulative-self.previous
            self.previous=cumulative
            # Word2Vec limpia norms al final de train(), no entre callbacks.
            # Recalcular evita cosenos con normas de una epoch anterior.
            model.wv.fill_norms(force=True)
            evaluation=epoch_evaluation(model.wv,selection_words,device)
            import psutil
            self.logs.append(dict(epoch=len(self.logs)+1,config=cfg,loss_sum=loss,
                                 training_seconds=training_seconds,gpu_peak_bytes=None,
                                 process_rss_bytes=psutil.Process().memory_info().rss,**evaluation))
            model.wv.save(str(CHECKPOINTS/f'gensim_e{len(self.logs)}.kv'),separately=[])
            save_json(ART/"runs"/"gensim.json",self.logs)
            print("GENSIM",len(self.logs),"rho",evaluation["wordsim"]["spearman"],flush=True)
    recorder=Recorder()
    model.train(sentences,total_examples=model.corpus_count,epochs=cfg["epochs"],
                compute_loss=True,callbacks=[recorder])
    model.wv.save(str(final))
    return model.wv,recorder.logs

PERSONAL_ANALOGIES=[
 ("gender","man","woman","king","queen"),("gender","boy","girl","father","mother"),
 ("gender","brother","sister","uncle","aunt"),
 ("capital","france","paris","germany","berlin"),("capital","italy","rome","japan","tokyo"),
 ("capital","spain","madrid","england","london"),
 ("nationality","france","french","italy","italian"),("nationality","germany","german","japan","japanese"),
 ("nationality","spain","spanish","england","english"),
 ("comparative","good","better","bad","worse"),("comparative","small","smaller","large","larger"),
 ("comparative","fast","faster","slow","slower"),
 ("past","walk","walked","play","played"),("past","go","went","run","ran"),
 ("past","eat","ate","write","wrote"),
 ("plural","cat","cats","dog","dogs"),("plural","book","books","car","cars"),
 ("plural","child","children","man","men")]

def final_embedding_evaluation(models,metas,device):
    frequencies=load_json(ART/"frequencies_selected.json")
    shared=set.intersection(*(set(kv.index_to_key) for kv in models.values()))
    vocabulary=sorted(shared,key=lambda w:(-frequencies.get(w,0),w))[:30000]
    if len(vocabulary)!=30000:
        raise RuntimeError(f"Solo hay {len(vocabulary)} palabras compartidas: se requieren 30000.")
    save_json(ART/"shared_vocabulary.json",vocabulary)
    questions=read_analogies()
    final,personal,parallelism={},{},{}
    for name,kv in models.items():
        engine=AnalogyEngine(kv,device,vocabulary)
        add=engine.benchmark(questions,"3CosAdd")
        mul=engine.benchmark(questions,"3CosMul")
        final[name]=dict(three_cos_add=add,three_cos_mul=mul,
                        wordsim=similarity_score(kv,read_similarity()),
                        simlex=similarity_score(kv,read_similarity("simlex999.txt")),
                        dimension=kv.vector_size,vocabulary_size=len(kv))
        del engine
        engine=AnalogyEngine(kv,device)
        records=[]
        for category,a,b,c,d in PERSONAL_ANALOGIES:
            item=dict(category=category,a=a,b=b,c=c,expected=d)
            if any(w not in kv for w in [a,b,c,d]):
                item.update(oov=[w for w in [a,b,c,d] if w not in kv],top5=[],without_exclusion=[],rank=None,cosine=None)
            else:
                top=engine.analogia(a,b,c)
                reference=kv.most_similar(positive=[b,c],negative=[a],topn=5)
                assert [w for w,_ in top]==[w for w,_ in reference],(name,a,b,c,top,reference)
                assert np.allclose([s for _,s in top],[s for _,s in reference],atol=2e-5)
                query=F.normalize(engine.query_vector(a,b,c),dim=0)
                scores=engine.vectors@query
                cosine=float(scores[engine.index[d]])
                for w in {a,b,c}:
                    scores[engine.index[w]]=-torch.inf
                rank=int((scores>cosine).sum())+1
                difference1=kv[b]-kv[a]
                difference2=kv[d]-kv[c]
                cosine_d=float(np.dot(difference1,difference2)/(np.linalg.norm(difference1)*np.linalg.norm(difference2)))
                item.update(top5=top,without_exclusion=engine.analogia(a,b,c,exclude=False),
                            rank=rank,cosine=cosine,parallel_cosine=cosine_d,oov=[])
            records.append(item)
        personal[name]=records
        by_category={category:[r["parallel_cosine"] for r in records if r["category"]==category and not r["oov"]]
                     for category in dict.fromkeys(r["category"] for r in records)}
        values=[r["parallel_cosine"] for r in records if not r["oov"]]
        parallelism[name]=dict(mean_cosine=float(np.mean(values)) if values else None,
                             evaluated=len(values),by_category={c:float(np.mean(v)) if v else None for c,v in by_category.items()})
        print("FINAL_EMBEDDING",name,final[name],flush=True)
        del engine
    save_json(ART/"evaluations"/"intrinsic.json",final)
    save_json(ART/"evaluations"/"personal_analogies.json",personal)
    save_json(ART/"evaluations"/"parallelism.json",parallelism)
    return final,personal,parallelism
