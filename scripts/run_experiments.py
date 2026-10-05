"""Ejecuta o reanuda el laboratorio; nunca selecciona usando test o SimLex."""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.common import initialize,ART,CHECKPOINTS,save_json,load_json,hardware,experiment_configs
from src.preprocessing import prepare_corpus,prepare_ids
from src.embeddings import load_glove,train_sgns,select_best,train_gensim,final_embedding_evaluation,read_similarity
from src.classification import prepare_news,train_classifiers,evaluate_test

def main(stage="all"):
    started=time.perf_counter()
    device=initialize()
    save_json(ART/"hardware.json",hardware())
    configs=experiment_configs()
    save_json(ART/"configurations.json",configs)
    corpus=prepare_corpus()
    metas={cfg["name"]:prepare_ids(cfg,corpus) for cfg in configs}
    save_json(ART/"corpus_configurations.json",list(metas.values()))
    if stage=="prepare":
        return
    glove=load_glove()
    # Candidatos fijos para comparar todas las iteraciones sin cambiar las
    # preguntas cubiertas. La evaluación FINAL usa el vocabulario de 3 modelos.
    common=set(glove.index_to_key)
    for meta in metas.values():
        common&=set(meta["vocabulary"])
    frequencies=load_json(ART/"frequencies_selected.json")
    selection_words=sorted(common,key=lambda w:(-frequencies.get(w,0),w))[:30000]
    save_json(ART/"selection_vocabulary.json",selection_words)
    save_json(ART/"selection_wordsim.json",[p for p in read_similarity() if p[0] in common and p[1] in common])
    for cfg in configs:
        train_sgns(cfg,metas[cfg["name"]],selection_words,device)
    best,sgns=select_best(configs)
    gensim,gensim_log=train_gensim(best,metas[best["config"]["name"]],selection_words,device)
    models=dict(sgns=sgns,gensim=gensim,glove=glove)
    # SimLex queda fuera del proceso de selección; se abre al cerrar SGNS.
    if not (ART/"evaluations"/"intrinsic.json").exists():
        final_embedding_evaluation(models,metas,device)
    if stage=="embeddings":
        return
    news=prepare_news(models)
    selection=train_classifiers(news,models,device)
    test=evaluate_test(news,selection,device)
    from src.reporting import tables,figures,tsne
    tsne(models)
    figures(tables())
    import importlib.metadata as metadata
    versions={d.metadata["Name"]:d.version for d in metadata.distributions()}
    save_json(ART/"environment.json",versions)
    save_json(ART/"completion.json",dict(complete=True,seconds_current_invocation=time.perf_counter()-started,
              sgns_iterations=len(configs),sgns_epochs=sum(len(load_json(ART/"runs"/(c["name"]+".json"))) for c in configs),
              classifiers=len(selection["all_results"]),final_test_models=len(test),seed=23236))
    print("LABORATORIO_COMPLETO",load_json(ART/"completion.json"),flush=True)

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--stage",choices=["all","prepare","embeddings"],default="all")
    main(parser.parse_args().stage)
