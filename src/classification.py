"""Clasificación AG News: selección con validación y test final bloqueado."""
from __future__ import annotations
import collections
import copy
import pickle
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, log_loss
from sklearn.model_selection import train_test_split
from .common import ART, DATA, SEED, save_json, load_json, sha256
from .preprocessing import tokenize

LABELS=["World","Sports","Business","Sci/Tech"]
FRACTIONS=[0.01,0.1,0.5,1.0]
NEWS_REVISION="eb185aade064a813bc0b7f42de02595523103ca4"

def metrics(y,prediction):
    p,r,f,_=precision_recall_fscore_support(y,prediction,average="macro",zero_division=0)
    return dict(accuracy=float(accuracy_score(y,prediction)),precision_macro=float(p),
                recall_macro=float(r),f1_macro=float(f))

def prepare_news(models):
    news=load_dataset("fancyzhx/ag_news",revision=NEWS_REVISION,cache_dir=str(DATA/"hf"))
    texts=news["train"]["text"]
    labels=np.array(news["train"]["label"],dtype=np.int64)
    train,validation=train_test_split(np.arange(len(texts)),test_size=0.1,stratify=labels,random_state=SEED)
    assert not set(train)&set(validation)
    tokens=[tokenize(text) for text in texts]
    test_texts=news["test"]["text"]
    test_tokens=[tokenize(text) for text in test_texts]
    test_labels=np.array(news["test"]["label"],dtype=np.int64)
    # Solo se mide cobertura de test; sus etiquetas no llegan al entrenamiento.
    frequencies=collections.Counter(w for i in train for w in tokens[i])
    test_frequencies=collections.Counter(w for doc in test_tokens for w in doc)
    validation_frequencies=collections.Counter(w for i in validation for w in tokens[i])
    oov={name:{split:dict(tokens=sum(counts.values()),unknown_tokens=sum(c for w,c in counts.items() if w not in kv),
                              unknown_fraction=sum(c for w,c in counts.items() if w not in kv)/sum(counts.values()))
               for split,counts in [("train",frequencies),("validation",validation_frequencies),("test",test_frequencies)]}
         for name,kv in models.items()}
    class_indices={}
    for label in range(4):
        indices=train[labels[train]==label].copy()
        np.random.default_rng(SEED+label).shuffle(indices)
        class_indices[label]=indices
    subsets={str(fraction):np.concatenate([ids[:max(1,int(len(ids)*fraction))] for ids in class_indices.values()])
             for fraction in FRACTIONS}
    save_json(ART/"news_split.json",dict(seed=SEED,train_indices=train.tolist(),validation_indices=validation.tolist(),
               official_test_size=len(test_labels),subsets={f:ids.tolist() for f,ids in subsets.items()},
               train_size=len(train),validation_size=len(validation),oov=oov,
               train_label_counts=np.bincount(labels[train],minlength=4).tolist(),
               validation_label_counts=np.bincount(labels[validation],minlength=4).tolist()))
    return dict(texts=texts,tokens=tokens,labels=labels,train=train,validation=validation,subsets=subsets,
                test_texts=test_texts,test_tokens=test_tokens,test_labels=test_labels,oov=oov)

def classifier_vocabulary(tokens):
    counts=collections.Counter(w for doc in tokens for w in doc)
    words=sorted((w for w,c in counts.items() if c>=2),key=lambda w:(-counts[w],w))[:50000]
    return ["<pad>","<unk>"]+words

def encode(tokens,vocabulary):
    index={w:i for i,w in enumerate(vocabulary)}
    return [np.array([index.get(w,1) for w in doc] or [1],dtype=np.int64) for doc in tokens]

class NewsClassifier(nn.Module):
    """EmbeddingBag con offsets evita padding y promedia una bolsa por noticia."""
    def __init__(self,weights,freeze=False):
        super().__init__()
        self.embedding=nn.EmbeddingBag.from_pretrained(weights,freeze=freeze,mode="mean",padding_idx=0)
        self.mlp=nn.Sequential(nn.Linear(weights.shape[1],128),nn.ReLU(),nn.Dropout(0.2),nn.Linear(128,4))
    def forward(self,flat_ids,offsets):
        return self.mlp(self.embedding(flat_ids,offsets))

def collate(encoded,indices,device):
    documents=[encoded[int(i)] for i in indices]
    offsets=np.cumsum([0]+[len(doc) for doc in documents[:-1]],dtype=np.int64)
    return torch.as_tensor(np.concatenate(documents),device=device),torch.as_tensor(offsets,device=device)

def predict_neural(model,encoded,labels,device):
    model.eval()
    prediction,loss_sum=[],0.0
    with torch.no_grad():
        for start in range(0,len(encoded),512):
            indices=np.arange(start,min(start+512,len(encoded)))
            ids,offsets=collate(encoded,indices,device)
            logits=model(ids,offsets)
            loss_sum+=float(F.cross_entropy(logits,torch.as_tensor(labels[indices],device=device),reduction="sum"))
            prediction.extend(logits.argmax(dim=1).cpu().tolist())
    return np.array(prediction),loss_sum/len(encoded)

def train_neural(name,initialization,freeze,fraction,news,kv,device):
    folder=ART/"classifiers"
    folder.mkdir(parents=True,exist_ok=True)
    key=f'{name}_{int(fraction*100):03d}'
    result_path=ART/"runs"/(key+".json")
    if result_path.exists() and (folder/(key+".pt")).exists():
        return load_json(result_path)
    subset=news["subsets"][str(fraction)]
    train_tokens=[news["tokens"][i] for i in subset]
    val_tokens=[news["tokens"][i] for i in news["validation"]]
    y=news["labels"][subset]
    yv=news["labels"][news["validation"]]
    vocabulary=classifier_vocabulary(train_tokens)
    dimension=100 if kv is None else kv.vector_size
    torch.manual_seed(SEED)
    weights=torch.empty(len(vocabulary),dimension).uniform_(-0.5/dimension,0.5/dimension)
    weights[0].zero_()
    # Las palabras fuera del embedding preentrenado parten de cero. En FT
    # pueden adquirir representación; congeladas no aportan contenido léxico.
    if kv is not None:
        weights.zero_()
        for i,word in enumerate(vocabulary[2:],2):
            if word in kv:
                weights[i]=torch.from_numpy(np.array(kv[word],copy=True))
    model=NewsClassifier(weights,freeze).to(device)
    optimizer=torch.optim.Adam((p for p in model.parameters() if p.requires_grad),lr=1e-3)
    train_encoded,val_encoded=encode(train_tokens,vocabulary),encode(val_tokens,vocabulary)
    history,best,best_epoch,bad=[],None,0,0
    started=time.perf_counter()
    for epoch in range(1,11):
        epoch_started=time.perf_counter()
        model.train()
        order=np.random.default_rng(SEED+epoch).permutation(len(subset))
        total_loss=0.0
        for start in range(0,len(order),512):
            indices=order[start:start+512]
            ids,offsets=collate(train_encoded,indices,device)
            target=torch.as_tensor(y[indices],device=device)
            optimizer.zero_grad(set_to_none=True)
            logits=model(ids,offsets)
            loss=F.cross_entropy(logits,target)
            loss.backward()
            optimizer.step()
            total_loss+=float(loss.detach())*len(indices)
        predictions,validation_loss=predict_neural(model,val_encoded,yv,device)
        validation=metrics(yv,predictions)
        history.append(dict(epoch=epoch,train_loss=total_loss/len(subset),validation_loss=validation_loss,
                            seconds=time.perf_counter()-epoch_started,**validation))
        if best is None or validation["f1_macro"]>best["f1_macro"]+1e-6:
            best,best_epoch,bad=validation,epoch,0
            torch.save(dict(state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                            vocabulary=vocabulary,dimension=dimension,freeze=freeze,epoch=epoch),folder/(key+".pt"))
        else:
            bad+=1
        print("CLASSIFIER",key,epoch,"val_f1",validation["f1_macro"],flush=True)
        if bad>=2:
            break
    result=dict(key=key,name=name,initialization=initialization,freeze=freeze,fraction=fraction,
                train_examples=len(subset),dimension=dimension,vocabulary_size=len(vocabulary),best_epoch=best_epoch,
                validation=best,history=history,training_seconds=time.perf_counter()-started,
                total_parameters=sum(p.numel() for p in model.parameters()),
                trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                checkpoint=str((folder/(key+".pt")).relative_to(ART)),
                checkpoint_sha256=sha256(folder/(key+".pt")))
    save_json(result_path,result)
    return result

def train_tfidf(fraction,news):
    folder=ART/"classifiers"
    folder.mkdir(parents=True,exist_ok=True)
    key=f'tfidf_{int(fraction*100):03d}'
    result_path=ART/"runs"/(key+".json")
    if result_path.exists() and (folder/(key+".pkl")).exists():
        return load_json(result_path)
    subset=news["subsets"][str(fraction)]
    y,yv=news["labels"][subset],news["labels"][news["validation"]]
    # El estimador sigue siendo regresión logística: se optimiza con SGD y
    # partial_fit permite registrar una curva auténtica por pasada de datos.
    vectorizer=TfidfVectorizer(tokenizer=tokenize,token_pattern=None,lowercase=False,
                               ngram_range=(1,2),min_df=2,max_features=100000,dtype=np.float32,sublinear_tf=True)
    started=time.perf_counter()
    x=vectorizer.fit_transform([news["texts"][i] for i in subset])
    xv=vectorizer.transform([news["texts"][i] for i in news["validation"]])
    model=SGDClassifier(loss="log_loss",alpha=1e-5,penalty="l2",average=True,random_state=SEED)
    history,best,best_epoch,bad=[],None,0,0
    for epoch in range(1,11):
        epoch_started=time.perf_counter()
        order=np.random.default_rng(SEED+epoch).permutation(len(subset))
        model.partial_fit(x[order],y[order],classes=np.arange(4))
        probability=model.predict_proba(xv)
        validation=metrics(yv,probability.argmax(axis=1))
        history.append(dict(epoch=epoch,train_loss=float(log_loss(y,model.predict_proba(x),labels=np.arange(4))),
                            validation_loss=float(log_loss(yv,probability,labels=np.arange(4))),
                            seconds=time.perf_counter()-epoch_started,**validation))
        if best is None or validation["f1_macro"]>best["f1_macro"]+1e-6:
            best,best_epoch,bad=validation,epoch,0
            with (folder/(key+".pkl")).open("wb") as f:
                pickle.dump((vectorizer,copy.deepcopy(model)),f)
        else:
            bad+=1
        print("CLASSIFIER",key,epoch,"val_f1",validation["f1_macro"],flush=True)
        if bad>=2:
            break
    result=dict(key=key,name="tfidf",initialization="tfidf",freeze=None,fraction=fraction,
                train_examples=len(subset),dimension=None,vocabulary_size=len(vectorizer.vocabulary_),best_epoch=best_epoch,
                validation=best,history=history,training_seconds=time.perf_counter()-started,
                total_parameters=int(model.coef_.size+model.intercept_.size),
                trainable_parameters=int(model.coef_.size+model.intercept_.size),
                checkpoint=str((folder/(key+".pkl")).relative_to(ART)),
                checkpoint_sha256=sha256(folder/(key+".pkl")))
    save_json(result_path,result)
    return result

def train_classifiers(news,models,device):
    all_results=[]
    for fraction in FRACTIONS:
        all_results.append(train_tfidf(fraction,news))
        all_results.append(train_neural("random","random",False,fraction,news,None,device))
        for name,kv in models.items():
            for freeze in [True,False]:
                all_results.append(train_neural(name+("_frozen" if freeze else "_finetune"),name,freeze,fraction,news,kv,device))
    selected=[]
    for fraction in FRACTIONS:
        for initialization in ["tfidf","random","sgns","gensim","glove"]:
            candidates=[r for r in all_results if r["fraction"]==fraction and r["initialization"]==initialization]
            selected.append(max(candidates,key=lambda r:(r["validation"]["f1_macro"],-(r["trainable_parameters"]))))
    # El manifiesto es escrito ANTES de leer etiquetas para evaluar test.
    lock=dict(selection_criterion="validation macro F1",selection_complete=True,
              selected=selected,all_results=all_results)
    path=ART/"classification_selection.json"
    if path.exists():
        previous=load_json(path)
        assert [r["checkpoint_sha256"] for r in previous["selected"]]==[r["checkpoint_sha256"] for r in selected]
    else:
        save_json(path,lock)
    return lock

def evaluate_test(news,selection,device):
    results=[]
    for row in selection["selected"]:
        path=ART/"evaluations"/("test_"+row["key"]+".json")
        checkpoint=ART/row["checkpoint"]
        assert sha256(checkpoint)==row["checkpoint_sha256"]
        if path.exists():
            result=load_json(path)
            assert result["checkpoint_sha256"]==row["checkpoint_sha256"]
            results.append(result)
            continue
        started=time.perf_counter()
        if row["initialization"]=="tfidf":
            with checkpoint.open("rb") as f:
                vectorizer,model=pickle.load(f)
            prediction=model.predict(vectorizer.transform(news["test_texts"]))
        else:
            state=torch.load(checkpoint,map_location="cpu",weights_only=False)
            weights=state["state"]["embedding.weight"]
            model=NewsClassifier(weights,state["freeze"]).to(device)
            model.load_state_dict(state["state"])
            encoded=encode(news["test_tokens"],state["vocabulary"])
            prediction,_=predict_neural(model,encoded,news["test_labels"],device)
        np.savetxt(ART/"evaluations"/("predictions_"+row["key"]+".csv"),
                   np.column_stack([np.arange(len(prediction)),news["test_labels"],prediction]),
                   delimiter=",",header="example_id,label,prediction",comments="",fmt="%d")
        result=dict(key=row["key"],initialization=row["initialization"],name=row["name"],fraction=row["fraction"],
                    checkpoint_sha256=row["checkpoint_sha256"],test_evaluations=1,
                    metrics=metrics(news["test_labels"],prediction),
                    confusion_matrix=confusion_matrix(news["test_labels"],prediction,labels=np.arange(4)).tolist(),
                    seconds=time.perf_counter()-started)
        save_json(path,result)
        results.append(result)
        print("TEST_FINAL",row["key"],result["metrics"],flush=True)
    save_json(ART/"evaluations"/"classification_test.json",results)
    return results
