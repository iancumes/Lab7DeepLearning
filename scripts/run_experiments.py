"""Ejecuta o reanuda el laboratorio; nunca selecciona usando test o SimLex."""
from __future__ import annotations
import argparse
import sys
import time
import shutil
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.common import ROOT,initialize,ART,CHECKPOINTS,save_json,load_json,hardware,experiment_configs
from src.preprocessing import prepare_corpus,prepare_ids
from src.embeddings import load_glove,train_sgns,select_best,train_gensim,final_embedding_evaluation,read_similarity
from src.classification import prepare_news,train_classifiers,evaluate_test

def archive_training_results():
    """Preserva resultados publicados antes de un entrenamiento nuevo."""
    archive=ROOT/'tmp'/f'resultados_previos_{time.time_ns()}'
    archive.mkdir(parents=True)
    sources=[ART/name for name in ['runs','evaluations','classifiers','classification_selection.json',
                                  'best_sgns.json','completion.json','verification.json']]
    sources+=list(CHECKPOINTS.glob('*.pt'))
    sources+=[p for p in CHECKPOINTS.glob('*.kv*') if not p.name.startswith('glove')]
    for source in sources:
        if not source.exists(): continue
        relative=source.relative_to(ROOT); destination=archive/relative
        assert source.resolve().is_relative_to(ROOT.resolve()) and destination.resolve().is_relative_to(ROOT.resolve())
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.move(str(source),str(destination))
    print('RESULTADOS_ANTERIORES_PRESERVADOS',archive,flush=True)

def main(stage="all",fresh=False):
    started=time.perf_counter()
    configs=experiment_configs()
    inherited=False
    if stage!='prepare':
        for cfg in configs:
            path=ART/'runs'/f"{cfg['name']}.json"
            if not path.exists(): continue
            rows=load_json(path)
            if rows and (rows[-1]['config']!=cfg or not (CHECKPOINTS/f"{cfg['name']}_e{rows[-1]['epoch']}.pt").exists()):
                inherited=True
    if fresh or inherited: archive_training_results()
    device=initialize()
    save_json(ART/"hardware.json",hardware())
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
    parser.add_argument('--fresh',action='store_true',help='Preservar resultados anteriores y entrenar desde cero.')
    args=parser.parse_args()
    main(args.stage,args.fresh)
