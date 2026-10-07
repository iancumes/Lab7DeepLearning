"""Discusión calculada desde métricas observadas, sin seleccionar con test."""
from .common import ART,load_json

def discussion():
    configs=load_json(ART/'configurations.json')
    runs={c['name']:load_json(ART/'runs'/f"{c['name']}.json") for c in configs}
    intrinsic=load_json(ART/'evaluations'/'intrinsic.json')
    tests=load_json(ART/'evaluations'/'classification_test.json')
    selection=load_json(ART/'classification_selection.json')
    personal=load_json(ART/'evaluations'/'personal_analogies.json')
    pct=lambda x:f'{100*x:.2f}%'
    out=['### Interpretación de los resultados observados']
    out.append('La curva log-log de frecuencias muestra la concentración de Zipf: pocas palabras muy frecuentes y una cola larga. No se estimó un exponente universal ni se probó un ajuste estadístico; el patrón visual y las coberturas respaldan su uso descriptivo.')
    ranked=sorted([row for rows in runs.values() for row in rows],key=lambda row:(row['wordsim']['spearman'] if row['wordsim']['spearman'] is not None else -2,row['analogies']['total']['accuracy'] or 0),reverse=True)
    first,second=ranked[:2]
    gap=first['wordsim']['spearman']-second['wordsim']['spearman']
    out.append(f"La selección sin redondear eligió {first['config']['name']} epoch {first['epoch']} (ρ={first['wordsim']['spearman']:.8f}) frente a {second['config']['name']} epoch {second['epoch']} (ρ={second['wordsim']['spearman']:.8f}); diferencia {gap:.8f}. Accuracy solo desempata correlaciones iguales. Una diferencia pequeña con una semilla no demuestra superioridad robusta.")
    for model,result in intrinsic.items():
        covered=[c for c in result['three_cos_add']['categories'] if c['evaluated']>0]
        good=max(covered,key=lambda c:c['accuracy']); bad=min(covered,key=lambda c:c['accuracy'])
        total=result['three_cos_add']['total']
        out.append(f"**{model}:** 3CosAdd total {pct(total['accuracy'])}; semánticas {pct(result['three_cos_add']['semantic']['accuracy'])}, sintácticas {pct(result['three_cos_add']['syntactic']['accuracy'])}. La categoría más fuerte fue {good['category']} ({pct(good['accuracy'])}) y la más débil {bad['category']} ({pct(bad['accuracy'])}). Se usan las mismas preguntas cubiertas y candidatos.")
    king=next(r for r in personal['sgns'] if (r['a'],r['b'],r['c'])==('man','woman','king'))
    out.append(f"En man:woman::king, queen tiene rango {king['rank']} y coseno {king['cosine']:.3f}. Al permitir consultas, el primer resultado es {king['without_exclusion'][0][0]}. El ≈ describe una dirección aproximada, no una identidad exacta ni garantía de recuperar queen. El paralelismo complementa el ranking, pero tampoco garantiza el acierto.")
    for name in ['dim50','dim300','corpus25','corpus50','window2','negative10']:
        row=runs[name][-1]; base=runs['base100'][-1]
        out.append(f"A epoch 3, **{name}**: WordSim {row['wordsim']['spearman']:.3f} frente a {base['wordsim']['spearman']:.3f} de base100; analogías {pct(row['analogies']['total']['accuracy'])} frente a {pct(base['analogies']['total']['accuracy'])}. Semánticas {pct(row['analogies']['semantic']['accuracy'])}, sintácticas {pct(row['analogies']['syntactic']['accuracy'])}. Tiempo total {sum(r['training_seconds'] for r in runs[name])/60:.1f} minutos.")
    w=runs['window2'][-1]; b=runs['base100'][-1]
    ds=w['analogies']['syntactic']['accuracy']-b['analogies']['syntactic']['accuracy']
    dm=w['analogies']['semantic']['accuracy']-b['analogies']['semantic']['accuracy']
    out.append(f"Reducir ventana 5→2 cambió accuracy sintáctica {100*ds:+.2f} puntos y semántica {100*dm:+.2f} puntos. Esto describe esta ejecución; no demuestra que ventanas pequeñas siempre favorezcan sintaxis. Con corpus 25/50/100% cambia además el vocabulario min_count y la frecuencia, aun usando preguntas comunes: se prueba la escala del pipeline completo.")
    s=intrinsic['sgns']['three_cos_add']['total']['accuracy']; g=intrinsic['glove']['three_cos_add']['total']['accuracy']; ge=intrinsic['gensim']['three_cos_add']['total']['accuracy']
    out.append(f"En evaluación final compartida, GloVe frente a SGNS: {100*(g-s):+.2f} puntos de accuracy; gensim frente a SGNS: {100*(ge-s):+.2f}. El experimento de corpus ofrece evidencia de escala dentro de SGNS, pero no permite repartir causalmente la diferencia con GloVe entre corpus y método. GloVe usa 6 mil millones de tokens y otra optimización. Gensim actualiza secuencialmente con cuatro workers, mientras SGNS suma gradientes de 4096 pares; cambia también el tratamiento de colisiones de negativos y el orden de muestreo.")
    out.append('Una pérdida menor no asegura mejores embeddings: SGNS separa pares reales y ruido; WordSim y analogías miden relaciones específicas. Aumentar k cambia el número de términos de la pérdida; no es válido ordenar k=5 y k=10 por su valor bruto.')
    counterexamples=[(name,a,b) for name,rows in runs.items() for a,b in zip(rows,rows[1:]) if b['loss']>a['loss'] and b['wordsim']['spearman']>a['wordsim']['spearman']]
    if counterexamples:
        name,a,b=counterexamples[0]
        out.append(f"Ejemplo observado dentro del mismo objetivo: {name}, epochs {a['epoch']}→{b['epoch']}, pérdida {a['loss']:.4f}→{b['loss']:.4f}, mientras WordSim mejora {a['wordsim']['spearman']:.4f}→{b['wordsim']['spearman']:.4f}. Por tanto, menor pérdida tampoco ordena necesariamente la calidad entre epochs de una misma configuración.")
    for fraction in [.01,.1,.5,1.0]:
        values={r['initialization']:r for r in tests if r['fraction']==fraction}
        chosen=[r for r in selection['selected'] if r['fraction']==fraction]
        variants=', '.join(r['name'] for r in chosen if r['initialization'] in ['sgns','gensim','glove'])
        metrics=', '.join(f"{m} {pct(values[m]['metrics']['f1_macro'])}" for m in ['tfidf','random','sgns','gensim','glove'])
        out.append(f"Con {100*fraction:g}% de etiquetas, F1 macro test: {metrics}. Validación eligió {variants}. GloVe frente a aleatorio: {100*(values['glove']['metrics']['f1_macro']-values['random']['metrics']['f1_macro']):+.2f} puntos. Se describen estas diferencias después de cerrar decisiones, sin volver a elegir variantes.")
    ranking_test=sorted([row for row in tests if row['fraction']==1],key=lambda row:row['metrics']['f1_macro'],reverse=True)
    first_test,second_test=ranking_test[:2]
    out.append(f"Al 100%, los dos mayores F1 test observados fueron {first_test['name']} ({first_test['metrics']['f1_macro']:.8f}) y {second_test['name']} ({second_test['metrics']['f1_macro']:.8f}); diferencia {100*(first_test['metrics']['f1_macro']-second_test['metrics']['f1_macro']):.4f} puntos porcentuales. El orden observado no demuestra una ventaja robusta y no se usó para volver a seleccionar variantes.")
    out.append('t-SNE representa vecindades y permite inspeccionar agrupaciones léxicas con solapamientos; los temas se asignaron antes de proyectar. Algunas categorías son gramaticales y mezclan significados. No se interpretan ejes ni separaciones globales como una escala de calidad. Tres epochs, una semilla y las actualizaciones asíncronas de gensim limitan la generalización; no hay intervalos de confianza ni afirmaciones de significancia.')
    return '\n\n'.join(out)
