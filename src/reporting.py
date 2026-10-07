"""Tablas y figuras reconstruidas exclusivamente desde resultados guardados."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from .common import ART,CHECKPOINTS,SEED,load_json,save_json
from .embeddings import read_analogies,epoch_evaluation

COLORS={'tfidf':'#222222','random':'#B58B24','sgns':'#28629C','gensim':'#A64B20','glove':'#61752F'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                     'axes.spines.right':False,'axes.grid':True,'grid.alpha':.18,
                     'figure.dpi':150,'savefig.bbox':'tight'})

def save_figure(fig,name):
    fig.tight_layout()
    fig.savefig(ART/'figures'/f'{name}.png')
    fig.savefig(ART/'figures'/f'{name}.pdf')
    plt.close(fig)

def tables():
    rows=[]
    for cfg in load_json(ART/'configurations.json'):
        for row in load_json(ART/'runs'/f"{cfg['name']}.json"):
            rows.append(dict(configuration=cfg['name'],epoch=row['epoch'],dimension=cfg['dim'],
              window=cfg['window'],negatives=cfg['negatives'],fraction=cfg['fraction'],tokens=row['corpus_tokens'],
              vocabulary=row['vocabulary_size'],loss=row['loss'],pairs=row['pairs'],
              retained_tokens=row['retained_tokens'],seconds=row['training_seconds'],gpu_bytes=row['gpu_peak_bytes'],
              wordsim=row['wordsim']['spearman'],
              semantic=row['analogies']['semantic']['accuracy'],syntactic=row['analogies']['syntactic']['accuracy'],
              accuracy=row['analogies']['total']['accuracy'],coverage=row['analogies']['total']['coverage']))
    epochs=pd.DataFrame(rows)
    intrinsic=load_json(ART/'evaluations'/'intrinsic.json')
    test=load_json(ART/'evaluations'/'classification_test.json')
    selection=load_json(ART/'classification_selection.json')
    selected={r['key']:r for r in selection['selected']}
    cls=pd.DataFrame([dict(model=r['initialization'],variant=r['name'],fraction=r['fraction'],
       train_examples=selected[r['key']]['train_examples'],val_f1=selected[r['key']]['validation']['f1_macro'],
       best_epoch=selected[r['key']]['best_epoch'],seconds=selected[r['key']]['training_seconds'],
       parameters=selected[r['key']]['total_parameters'],trainable_parameters=selected[r['key']]['trainable_parameters'],
       **r['metrics']) for r in test])
    comparison=[]
    categories=[]
    best=load_json(ART/'best_sgns.json')
    sgns_runs=load_json(ART/'runs'/f"{best['config']['name']}.json")
    gensim_runs=load_json(ART/'runs'/'gensim.json')
    news=load_json(ART/'news_split.json')
    parallel=load_json(ART/'evaluations'/'parallelism.json')
    hardware=load_json(ART/'hardware.json')
    for model,r in intrinsic.items():
        final=cls[(cls.model==model)&(cls.fraction==1)].iloc[0]
        comparison.append(dict(model=model,dimension=r['dimension'],vocabulary=r['vocabulary_size'],
           semantic=r['three_cos_add']['semantic']['accuracy'],syntactic=r['three_cos_add']['syntactic']['accuracy'],
           analogy_add=r['three_cos_add']['total']['accuracy'],analogy_mul=r['three_cos_mul']['total']['accuracy'],
           analogy_coverage=r['three_cos_add']['total']['coverage'],wordsim=r['wordsim']['spearman'],
           wordsim_coverage=r['wordsim']['coverage'],simlex=r['simlex']['spearman'],
           simlex_coverage=r['simlex']['coverage'],test_f1=final.f1_macro,test_variant=final.variant))
        comparison[-1].update(corpus_tokens=6_000_000_000 if model=='glove' else best['corpus_tokens'],
            training_seconds=None if model=='glove' else sum(e['training_seconds'] for e in (sgns_runs if model=='sgns' else gensim_runs)),
            hardware='dual Intel Xeon E5-2658; 32 cores (artículo, 300d)' if model=='glove' else (str(hardware.get('gpu','CPU')) if model=='sgns' else str(hardware.get('cpu','CPU Colab'))),
            parallel_cosine=parallel[model]['mean_cosine'],ag_news_oov=news['oov'][model]['test']['unknown_fraction'],
            published_timing_note='85 min coocurrencias; 14 min/iteración para 300d, no 100d' if model=='glove' else 'medición de este experimento')
        for method in ['three_cos_add','three_cos_mul']:
            categories.extend(dict(model=model,method=method,**cat) for cat in r[method]['categories'])
    output=dict(epochs=epochs,classifiers=cls,comparison=pd.DataFrame(comparison),categories=pd.DataFrame(categories))
    for name,frame in output.items():
        frame.to_csv(ART/f'{name}.csv',index=False)
    return output

def figures(output=None):
    output=output or tables()
    epochs,cls=output['epochs'],output['classifiers']
    counts=np.array(sorted(load_json(ART/'frequencies_full_train.json').values(),reverse=True))
    fig,ax=plt.subplots(figsize=(7,3.5))
    ax.loglog(np.arange(1,len(counts)+1),counts,color=COLORS['sgns'])
    ax.set(xlabel='Rango de frecuencia (log)',ylabel='Frecuencia (log)',title='Ley de Zipf en WikiText 103 normalizado')
    save_figure(fig,'zipf')
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    styles={'base100':('#28629C','-','o'),'dim50':('#28629C','--','^'),'dim300':('#28629C',':','D'),
            'corpus25':('#61752F','--','^'),'corpus50':('#61752F',':','D'),
            'window2':('#A64B20','-','s'),'negative10':('#9A5278','-','v')}
    for name,group in epochs.groupby('configuration',sort=False):
        color,style,marker=styles[name]
        axes[0].plot(group.epoch,group.loss,marker=marker,color=color,linestyle=style,label=name)
        axes[1].plot(group.epoch,group.wordsim,marker=marker,color=color,linestyle=style,label=name)
    axes[0].set(xlabel='Epoch',ylabel='Pérdida media SGNS por par',xticks=[1,2,3])
    axes[1].set(xlabel='Epoch',ylabel='WordSim 353 Spearman',xticks=[1,2,3])
    axes[1].legend(fontsize=8,ncol=2)
    save_figure(fig,'sgns_curves')
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for name,group in epochs.groupby('configuration',sort=False):
        color,style,marker=styles[name]
        for ax,metric,label in zip(axes,['semantic','syntactic','accuracy'],['Semánticas','Sintácticas','Total']):
            ax.plot(group.epoch,group[metric]*100,marker=marker,color=color,linestyle=style,label=name)
            ax.set(xlabel='Epoch',ylabel=f'Accuracy {label.lower()} (%)',xticks=[1,2,3])
    axes[-1].legend(fontsize=8,ncol=2)
    save_figure(fig,'sgns_accuracy')
    group=epochs[epochs.configuration.isin(['corpus25','corpus50','base100'])].sort_values('tokens')
    best=group.sort_values(['wordsim','accuracy'],ascending=False).groupby('configuration',sort=False).head(1).sort_values('tokens')
    reference=ART/'glove_selection_evaluation.json'
    if not reference.exists():
        from gensim.models import KeyedVectors
        import torch
        kv=KeyedVectors.load(str(CHECKPOINTS/'glove.kv'),mmap='r')
        save_json(reference,epoch_evaluation(kv,load_json(ART/'selection_vocabulary.json'),torch.device('cuda' if torch.cuda.is_available() else 'cpu')))
    glove=load_json(reference)['analogies']['total']['accuracy']
    fig,ax=plt.subplots(figsize=(7,3.5))
    ax.plot(best.tokens/1e6,best.accuracy*100,'o-',color=COLORS['sgns'],label='SGNS checkpoint por WordSim')
    ax.axhline(glove*100,color=COLORS['glove'],linestyle='--',label='GloVe 100d (6 mil millones de tokens)')
    ax.set(xlabel='Millones de tokens normalizados del corpus',ylabel='Accuracy de analogías (%)')
    ax.legend(fontsize=8)
    save_figure(fig,'analogy_vs_tokens')
    fig,ax=plt.subplots(figsize=(7,3.7))
    for name,group in cls.groupby('model',sort=False):
        group=group.sort_values('fraction')
        ax.plot(group.fraction*100,group.f1_macro*100,marker='o',color=COLORS[name],label=name)
    ax.set(xlabel='Entrenamiento AG News disponible (%)',ylabel='F1 macro en test (%)',xticks=[1,10,50,100])
    ax.legend(ncol=3,fontsize=8)
    save_figure(fig,'classification_vs_data')
    chosen=load_json(ART/'classification_selection.json')['all_results']
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for row in chosen:
        if row['fraction']!=1: continue
        history=pd.DataFrame(row['history'])
        color=COLORS[row['initialization']]
        style='--' if row['freeze'] else '-'
        axes[0].plot(history.epoch,history.train_loss,label=row['name'],color=color,linestyle=style)
        axes[1].plot(history.epoch,history.validation_loss,label=row['name'],color=color,linestyle=style)
        axes[2].plot(history.epoch,history.f1_macro*100,label=row['name'],color=color,linestyle=style)
    axes[0].set(xlabel='Epoch',ylabel='Pérdida de entrenamiento')
    axes[1].set(xlabel='Epoch',ylabel='Pérdida de validación')
    axes[2].set(xlabel='Epoch',ylabel='F1 macro de validación (%)')
    axes[2].legend(fontsize=7,ncol=2)
    save_figure(fig,'classification_curves')
    tests=[r for r in load_json(ART/'evaluations'/'classification_test.json') if r['fraction']==1]
    fig,axes=plt.subplots(1,5,figsize=(16,3.5))
    for ax,row in zip(axes,tests):
        matrix=np.array(row['confusion_matrix'])
        ax.imshow(matrix,cmap='Blues',vmin=0,vmax=max(np.max(np.array(r['confusion_matrix'])) for r in tests))
        for y in range(4):
            for x in range(4): ax.text(x,y,str(matrix[y,x]),ha='center',va='center',fontsize=8,color='white' if matrix[y,x]>1000 else 'black')
        ax.set(title=row['name'],xticks=range(4),yticks=range(4),xticklabels=['W','S','B','T'],yticklabels=['W','S','B','T'],xlabel='Predicción')
        ax.grid(False)
    axes[0].set_ylabel('Etiqueta real')
    save_figure(fig,'confusion_matrices')

def tsne(models):
    shared=set(load_json(ART/'shared_vocabulary.json'))
    groups={k:set() for k in ['Geografía y monedas','Familia','Adjetivos','Verbos','Sustantivos']}
    for q in read_analogies():
        category=q['category']; words=q['words']
        if category=='family': group='Familia'
        elif not category.startswith('gram'): group='Geografía y monedas'
        elif category.startswith(('gram1','gram2','gram3','gram4')): group='Adjetivos'
        elif category.startswith(('gram5','gram7','gram9')): group='Verbos'
        elif category.startswith('gram6'): group='Geografía y monedas'
        elif category.startswith('gram8'): group='Sustantivos'
        else: continue
        groups[group].update(w for w in words if w in shared)
    chosen={}; used=set()
    # Muestreo equilibrado por rondas; las etiquetas proceden de la relación,
    # no de la posición t-SNE observada. Evita inferir grupos después del gráfico.
    order={g:iter(np.random.default_rng(SEED).permutation(sorted(words)).tolist()) for g,words in groups.items()}
    while len(chosen)<500:
        advanced=False
        for group,iterator in order.items():
            for word in iterator:
                if word not in used:
                    chosen[word]=group; used.add(word); advanced=True; break
            if len(chosen)==500: break
        if not advanced: break
    assert len(chosen)>=450,len(chosen)
    words=list(chosen)
    rows=[]
    fig,axes=plt.subplots(1,3,figsize=(15,4.5))
    palette=dict(zip(groups,['#28629C','#A64B20','#B58B24','#61752F','#9A5278']))
    markers=dict(zip(groups,['o','s','^','D','v']))
    for ax,(model,kv) in zip(axes,models.items()):
        matrix=np.array([kv.get_vector(w,norm=True) for w in words])
        coordinates=TSNE(n_components=2,perplexity=30,init='pca',metric='cosine',max_iter=1500,random_state=SEED,n_jobs=2).fit_transform(matrix)
        for group in groups:
            mask=np.array([chosen[w]==group for w in words])
            ax.scatter(coordinates[mask,0],coordinates[mask,1],s=13,alpha=.7,label=group,color=palette[group],marker=markers[group])
        for i in range(0,len(words),40): ax.annotate(words[i],coordinates[i],fontsize=8)
        ax.set(title=f'{model} ({len(words)} palabras)',xlabel='t SNE 1',ylabel='t SNE 2')
        rows.extend(dict(model=model,word=w,group=chosen[w],x=float(x),y=float(y)) for w,(x,y) in zip(words,coordinates))
    axes[-1].legend(fontsize=9)
    save_figure(fig,'tsne')
    pd.DataFrame(rows).to_csv(ART/'tsne_coordinates.csv',index=False)
    focus=pd.DataFrame([r for r in rows if r['model']=='sgns'])
    fig,ax=plt.subplots(figsize=(7.2,2.7))
    for group in groups:
        part=focus[focus.group==group]
        ax.scatter(part.x,part.y,s=10,alpha=.7,label=group,color=palette[group],marker=markers[group])
    for i in range(0,len(focus),45):
        r=focus.iloc[i]; ax.annotate(r.word,(r.x,r.y),fontsize=8)
    ax.set(xlabel='t SNE 1',ylabel='t SNE 2')
    ax.legend(fontsize=7,ncol=3,loc='upper center',bbox_to_anchor=(.5,1.35))
    save_figure(fig,'tsne_sgns')
    save_json(ART/'tsne_selection.json',dict(seed=SEED,words=chosen,perplexity=30,max_iter=1500,metric='cosine',note='Grupos léxicos definidos por categorías de analogías; ejes y distancias globales no tienen interpretación semántica directa.'))
