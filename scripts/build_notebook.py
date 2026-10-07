"""Construye el notebook completo; ejecutar después desde un kernel limpio."""
from pathlib import Path
import sys
import nbformat as nbf
ROOT=Path(__file__).resolve().parents[1]
nb=nbf.v4.new_notebook()
cells=[]
def md(text): cells.append(nbf.v4.new_markdown_cell(text.strip()))
def code(text): cells.append(nbf.v4.new_code_cell(text.strip()))
md(r'''# Laboratorio 7 — NLP y embeddings
**Ian Cumes · carné 23236 · entrega individual**

Este notebook reúne investigación, implementación SGNS propia, evaluación vectorial y clasificación AG News. Los resultados se leen de registros reales del entrenamiento. El código completo está en `src/`, los manifiestos y métricas en `artifacts/`; los vectores grandes se distribuyen en una release.

[Repositorio](https://github.com/iancumes/Lab7DeepLearning) · [Enunciado](https://github.com/iancumes/Lab7DeepLearning/blob/main/Laboratorio_7_NLP_Embeddings.md)

**Ejecución:** todas las celdas funcionan en un kernel nuevo con dependencias instaladas. En Colab se clona el repositorio y se instalan las versiones fijadas; seleccionar el runtime 2025.10 con Python 3.12. Si Colab solicita reinicio después de instalar, reiniciar el kernel y ejecutar todas las celdas. `REENTRENAR=True` ejecuta los experimentos completos o reanuda checkpoints presentes. Si solo hay métricas publicadas y faltan checkpoints, el script preserva esos resultados en `tmp/` e inicia un entrenamiento nuevo. La ejecución predeterminada reconstruye tablas y figuras desde las métricas publicadas, evitando repetir entrenamiento y test. Se necesitan Python 3.11/3.12 y varios GB de disco para reentrenar.''')
code('''from pathlib import Path
import os, sys, subprocess
EN_COLAB = 'google.colab' in sys.modules
if EN_COLAB:
    raiz = Path('/content/Lab7DeepLearning')
    if not raiz.exists():
        subprocess.run(['git','clone','https://github.com/iancumes/Lab7DeepLearning.git',str(raiz)],check=True)
    os.chdir(raiz)
    subprocess.run([sys.executable,'-m','pip','install','-q','-r','requirements.txt'],check=True)
else:
    raiz = Path.cwd()
    if raiz.name == 'notebooks': raiz = raiz.parent
    os.chdir(raiz)
sys.path.insert(0,str(raiz))
REENTRENAR = False
print('Repositorio:',raiz,'· reentrenar:',REENTRENAR)''')
code('''import json, inspect
import numpy as np
import pandas as pd
from IPython.display import display, Markdown, Image
from src.common import ART, CHECKPOINTS, load_json, experiment_configs
from src.preprocessing import tokenize, normalize, pair_batches, pair_count
from src.embeddings import SGNS, AnalogyEngine
from src.classification import NewsClassifier
from src.reporting import tables, figures
pd.set_option('display.max_rows',200)
pd.set_option('display.max_columns',20)
pd.set_option('display.max_colwidth',160)
if REENTRENAR:
    subprocess.run([sys.executable,'-u','scripts/run_experiments.py'],check=True)
assert (ART/'completion.json').exists(), 'Ejecutar primero los experimentos completos.'
print('Evidencia de ejecución:',load_json(ART/'completion.json'))
subprocess.run([sys.executable,'-m','pytest','-q'],check=True)''')
md(r'''## 1 Investigación y preprocesamiento

### Word2Vec y GloVe
CBOW predice la palabra central a partir de su contexto; Skip-Gram predice palabras del contexto desde el centro. SGNS reemplaza el softmax de todo el vocabulario por discriminación binaria entre pares observados y negativos. Las dos tablas cumplen papeles distintos: vectores de entrada del centro y vectores de salida del contexto. Se conserva la tabla de entrada para evaluación y clasificación.

GloVe minimiza un error ponderado sobre logaritmos de coocurrencias globales: $J=\sum_{ij}f(X_{ij})(w_i^T\tilde w_j+b_i+\tilde b_j-\log X_{ij})^2$. La ponderación limita la influencia de pares excesivamente frecuentes; el modelo incorpora estadísticas del corpus completo. Word2Vec actualiza ejemplos locales y GloVe usa una matriz global dispersa. Ambos aprenden relaciones distribucionales y pueden reflejar sesgos del corpus. No distinguen sentidos de una misma palabra ni conservan orden cuando se promedian.

[Mikolov et al. 2013](https://papers.nips.cc/paper_files/paper/2013/hash/9aa42b31882ec039965f3c4923ce901b-Abstract.html) · [Pennington et al. 2014](https://nlp.stanford.edu/pubs/glove.pdf)

### Capas de PyTorch
`nn.Embedding(V,d)` transforma IDs enteros en filas de una matriz entrenable. `padding_idx` evita actualizar esa fila; `sparse=True` genera gradientes dispersos compatibles con SGD. `from_pretrained(weights, freeze=True)` inicializa desde una matriz y congela sus pesos; `freeze=False` permite ajuste. `EmbeddingBag` reduce los embeddings de cada documento por `mean`, `sum` o `max` sin materializar padding. Para una lista de IDs concatenados, `offsets` señala el comienzo de cada documento; `include_last_offset=True` añade además el límite final. Aquí se usa `mean`, offsets iniciales y `padding_idx=0`.

[Embedding, PyTorch 2.8](https://docs.pytorch.org/docs/2.8/generated/torch.nn.Embedding.html) · [EmbeddingBag, PyTorch 2.8](https://docs.pytorch.org/docs/2.8/generated/torch.nn.EmbeddingBag.html)

### Normalización
Se usan minúsculas, comillas ASCII, reparación de `@-@`, `@,@`, `@.@`, eliminación de `<unk>`, `<pad>` y signos de encabezado. Se conservan números y puntuación, sin stemming ni stopwords. La expresión regular admite palabras Unicode y apóstrofos internos. Es compatible con GloVe uncased, aunque algunas contracciones y números tienen segmentación distinta. Un artículo empieza en un encabezado de nivel uno; una línea no equivale a un artículo.''')
code('''print(inspect.getsource(normalize))
print(inspect.getsource(tokenize))
ejemplos=pd.DataFrame(load_json(ART/'tokenizer_examples.json'))
ejemplos['iguales']=ejemplos.custom.map(tuple)==ejemplos.nltk.map(tuple)
display(ejemplos)''')
md(r'''La comparación usa `TreebankWordTokenizer` sobre la misma normalización para aislar diferencias de tokenización. NLTK separa contracciones como `can't` y `John's`; el tokenizador propio conserva apóstrofos internos. Ambos pueden fragmentar abreviaturas, URL y correos. No se fuerza igualdad: esas diferencias explican parte del OOV respecto de GloVe.''')
code('''corpus=load_json(ART/'corpus.json')
display(pd.DataFrame(corpus['splits']).T)
display(pd.DataFrame(corpus['thresholds']))
display(pd.DataFrame([{'top_k':int(k),'cobertura':v} for k,v in corpus['zipf_coverage'].items()]))
print('Artículos seleccionados:',corpus['selected_articles'])
print('Tokens normalizados seleccionados:',corpus['selected_tokens'])
print('SHA256 del corpus:',corpus['selected_sha256'])
assert corpus['selected_tokens']>=20_000_000
print('Primeros artículos del manifiesto:'); display(pd.DataFrame(corpus['articles']).head(10))''')
code('''from src.reporting import tables as construir_tablas
tablas=construir_tablas()
figures(tablas)
display(Image(filename=str(ART/'figures'/'zipf.png')))''')
md(r'''Zipf relaciona rango y frecuencia aproximadamente por una potencia inversa; la gráfica usa ejes logarítmicos. La concentración del corpus justifica subsampling: stopwords aportan muchos pares redundantes. `min_count` controla rareza, memoria y OOV; `min_count=1` conserva todos los tipos y 5/10 eliminan los menos observados. Los tokens excluidos por min_count no se entrenan como una palabra artificial `<unk>`.

Se permutan artículos completos con semilla 23236 hasta superar 20 millones de tokens normalizados. Los subconjuntos del 25 %, 50 % y 100 % son prefijos anidados de esa selección y cierran en límites de artículos. Por eso pueden exceder ligeramente su porcentaje nominal. Las líneas normalizadas se dividen en bloques de máximo 1000 tokens para evitar truncamiento de gensim; ambos entrenamientos usan exactamente el mismo archivo y límites de contexto.''')
code('''configuraciones=load_json(ART/'configurations.json')
display(pd.DataFrame(configuraciones))
metadata=[load_json(ART/'ids'/c['name']/'meta.json') for c in configuraciones] if (ART/'ids'/'base100'/'meta.json').exists() else load_json(ART/'corpus_configurations.json')
display(pd.DataFrame([{'configuración':c['name'],'tokens':m['corpus_tokens'],'tokens_elegibles':m['eligible_tokens'],'vocabulario':len(m['vocabulary']),'pares_antes':m['pairs_before_subsampling'],'sha256':m['corpus_sha256']} for c,m in zip(configuraciones,metadata)]))''')
md(r'''## 2 Entrenamiento SGNS propio y referencias

Para centro $w$ y contexto positivo $c$, se minimiza $-\log\sigma(v_w^Tu_c)-\sum_{j=1}^k\log\sigma(-v_w^Tu_{n_j})$. Los negativos se extraen de $P(n)\propto count(n)^{0.75}$ y se redibujan si coinciden con el contexto positivo. La probabilidad de conservar una palabra de frecuencia relativa $f$ es $\min(1,(\sqrt{f/t}+1)t/f)$ con $t=10^{-4}$.

Se generan pares ordenados por bloques de centros, con ventana fija ±w y sin cruzar líneas. Los tokens descartados se eliminan antes de calcular contexto. Se usan dos `nn.Embedding` con gradientes dispersos, SGD y learning rate lineal 0.025→0.0001 a lo largo de tres epochs. El minibatch SGNS es de 4096 pares en todas las configuraciones, elegido después de una prueba de estabilidad/rendimiento con pares reales que no participa en la selección. La pérdida se suma para actualizar contribuciones de pares y se reporta su media por par. Esto no reproduce la actualización secuencial de Word2Vec: los pares de un minibatch comparten el estado previo a la actualización. La inicialización de salida es cero y la de entrada uniforme ±0.5/d.

Las siete configuraciones cambian un factor respecto de la base: dimensión 50/300, corpus 25/50 %, ventana 2 y diez negativos. La selección de configuración y epoch usa WordSim-353 sobre pares cubiertos por todas las variantes, con accuracy de analogías para desempatar. Las analogías de selección usan candidatos fijos. SimLex-999 no se consulta durante esa selección.''')
code('''print(inspect.getsource(SGNS))
print(inspect.getsource(pair_batches))
display(tablas['epochs'].round(5))''')
code('''subs=[]
for cfg in configuraciones:
    for row in load_json(ART/'runs'/f"{cfg['name']}.json"):
        for word,stats in row['subsampling'].items():
            subs.append({'configuración':cfg['name'],'epoch':row['epoch'],'palabra':word,**stats})
display(pd.DataFrame(subs))
display(Image(filename=str(ART/'figures'/'sgns_curves.png')))
display(Image(filename=str(ART/'figures'/'sgns_accuracy.png')))''')
md(r'''La pérdida SGNS depende del número de negativos: los valores absolutos de k=5 y k=10 no son directamente comparables. La selección usa WordSim y analogías, no la menor pérdida entre objetivos diferentes.''')
code('''neighbors=[]
for cfg in configuraciones:
    for row in load_json(ART/'runs'/f"{cfg['name']}.json"):
        for word,values in row['neighbors'].items():
            neighbors.append({'configuración':cfg['name'],'epoch':row['epoch'],'palabra':word,'vecinos':', '.join(f'{w} ({s:.3f})' for w,s in values)})
display(pd.DataFrame(neighbors))
mejor=load_json(ART/'best_sgns.json')
print('Checkpoint elegido:',mejor['config']['name'],'epoch',mejor['epoch'],'WordSim',mejor['wordsim'])
display(pd.DataFrame([load_json(ART/'hardware.json')]))''')
md(r'''### Referencia gensim y GloVe
Gensim usa `sg=1`, `hs=0`, la misma dimensión, ventana fija (`shrink_windows=False`), min_count, subsampling, negativos y distribución 0.75; tiene tres epochs y las mismas tasas inicial/final. Se verifica igualdad del vocabulario y del archivo de corpus por SHA256. Su entrenamiento usa Cython y cuatro workers; las actualizaciones asíncronas, orden, implementación del muestreo y aprendizaje secuencial producen diferencias incluso con una semilla igual. Gensim descarta un negativo que coincide con el positivo, mientras esta implementación lo redibuja. Su pérdida acumulada se diferencia por epoch, pero no es directamente comparable con la media SGNS por par.

Se carga **exactamente `glove-wiki-gigaword-100`**, con 400,000 palabras, 100 dimensiones y entrenamiento publicado sobre 6 mil millones de tokens. No se reentrena GloVe en este laboratorio. El artículo original describe dos Intel Xeon E5-2658 de 2.1 GHz: 85 minutos para coocurrencias en un hilo y 14 minutos por iteración para vectores de **300 dimensiones** con 32 cores. Esos tiempos no son mediciones de `glove.6B.100d`, ni deben compararse como si fueran del mismo hardware/corpus/dimensión.

[Gensim Word2Vec](https://radimrehurek.com/gensim/models/word2vec.html) · [GloVe](https://nlp.stanford.edu/projects/glove/)''')
code('''gensim_log=load_json(ART/'runs'/'gensim.json')
display(pd.DataFrame([{k:v for k,v in r.items() if k not in ['neighbors','analogies','wordsim']} | {'wordsim':r['wordsim']['spearman'],'accuracy':r['analogies']['total']['accuracy']} for r in gensim_log]))
display(pd.DataFrame([load_json(ART/'glove_source.json')]))
display(pd.DataFrame([{'epoch':r['epoch'],'palabra':w,'vecinos':', '.join(f'{word} ({s:.3f})' for word,s in neighbors)} for r in gensim_log for w,neighbors in r['neighbors'].items()]))
display(Image(filename=str(ART/'figures'/'analogy_vs_tokens.png')))''')
md(r'''## 3 Aritmética vectorial y evaluación intrínseca

`analogia(a,b,c,k)` implementa **3CosAdd**: normaliza cada palabra y busca vecinos por coseno de $\hat b-\hat a+\hat c$. Se excluyen a, b y c, como `gensim.most_similar`; se verifica igualdad del top-5 y similitudes con tolerancia numérica. **3CosMul** desplaza cosenos a [0,1] y maximiza $(1+cos(x,b))(1+cos(x,c))/(2(1+cos(x,a))+\epsilon)$; la implementación equivalente usa productos de cosenos desplazados y epsilon 1e-6.

El benchmark tiene 19,544 preguntas y 14 categorías. Se exige que las cuatro palabras estén en los mismos 30,000 candidatos compartidos para todos los modelos. Accuracy se divide entre preguntas cubiertas, y cobertura entre todas las preguntas; un resultado alto con baja cobertura no significa éxito global. La columna de categorías permite distinguir relaciones semánticas y sintácticas. WordSim y SimLex finales reportan cobertura propia de cada modelo; WordSim utilizado para selección usa pares fijos.''')
md(r'''Spearman compara rangos de cosenos y puntuaciones humanas, sin asumir una relación lineal. WordSim-353 incluye asociaciones y relaciones de significado; SimLex-999 se diseñó para medir similitud, diferenciándola de asociación. Por eso las correlaciones pueden cambiar de un benchmark a otro. [SimLex-999](https://fh295.github.io/simlex.html).''')
code('''print(inspect.getsource(AnalogyEngine.analogia))
display(tablas['comparison'].round(5))
display(tablas['categories'].round(5))
intrinsic=load_json(ART/'evaluations'/'intrinsic.json')
assert len(load_json(ART/'shared_vocabulary.json'))==30000
assert len({r['three_cos_add']['total']['evaluated'] for r in intrinsic.values()})==1
print('Mismas preguntas cubiertas y 30000 candidatos en los tres embeddings.')''')
code('''from gensim.models import KeyedVectors
from src.common import sha256
import urllib.request
_motor_analogia=None
def analogia(a,b,c,k=5):
    """3CosAdd sobre el mejor SGNS; carga los vectores solo en la primera llamada."""
    global _motor_analogia
    if _motor_analogia is None:
        archivo=CHECKPOINTS/'best_sgns.kv'
        if not archivo.exists():
            archivo.parent.mkdir(exist_ok=True)
            urllib.request.urlretrieve('https://github.com/iancumes/Lab7DeepLearning/releases/download/v1.0-lab7/best_sgns.kv',archivo)
        manifiesto=ART/'vector_release.json'
        if manifiesto.exists(): assert sha256(archivo)==load_json(manifiesto)['sha256']
        _motor_analogia=AnalogyEngine(KeyedVectors.load(str(archivo)))
    return _motor_analogia.analogia(a,b,c,k)
display(pd.DataFrame(analogia('man','woman','king',5),columns=['palabra','coseno']))''')
code('''personal=load_json(ART/'evaluations'/'personal_analogies.json')
analogias=[]
for model,rows in personal.items():
    for r in rows:
        analogias.append({'modelo':model,'tipo':r['category'],'consulta':f"{r['a']} : {r['b']} :: {r['c']} : ?",'esperado':r['expected'],'rango':r['rank'],'coseno_correcto':r['cosine'],'top5':', '.join(f'{w} ({s:.3f})' for w,s in r['top5']),'sin_exclusión':', '.join(f'{w} ({s:.3f})' for w,s in r['without_exclusion']),'oov':r['oov']})
display(pd.DataFrame(analogias))''')
md(r'''Se evalúan 18 analogías en seis tipos: género, capitales, nacionalidades, comparativos, pasado y plurales. El rango correcto se calcula entre todos los candidatos del modelo, excluyendo las consultas. El top-5 sin exclusión muestra por qué una entrada como `king` puede dominar la búsqueda. La similitud con la respuesta esperada y su rango permiten distinguir una aproximación semántica de un acierto exacto.

El paralelismo se mide mediante $cos(b-a,d-c)$ para relaciones equivalentes. Usa diferencias de vectores originales y no implica necesariamente una analogía correcta: el ruido, la norma y competidores léxicos también influyen. La proyección t-SNE permite explorar vecindades, pero sus ejes, distancias globales y tamaños de grupos no tienen interpretación semántica directa.''')
code('''display(pd.DataFrame(load_json(ART/'evaluations'/'parallelism.json')).T)
for model,rows in personal.items():
    cubiertos=[r for r in rows if not r['oov']]
    aciertos=[r for r in cubiertos if r['rank']==1]
    fallos=sorted([r for r in cubiertos if r['rank']>1],key=lambda r:r['rank'],reverse=True)
    print(model,'aciertos top1:',len(aciertos),'/',len(cubiertos))
    if aciertos: print('Ejemplo de acierto:',aciertos[0]['a'],aciertos[0]['b'],aciertos[0]['c'],'→',aciertos[0]['expected'])
    if fallos: print('Ejemplo de fallo:',fallos[0]['a'],fallos[0]['b'],fallos[0]['c'],'esperado',fallos[0]['expected'],'top1',fallos[0]['top5'][0],'rango',fallos[0]['rank'])
display(Image(filename=str(ART/'figures'/'tsne.png')))
print(load_json(ART/'tsne_selection.json')['note'])''')
md(r'''## 4 Clasificación AG News

Se separa el entrenamiento oficial estratificadamente en 108,000 ejemplos de entrenamiento y 12,000 de validación. El test oficial tiene 7,600 ejemplos. Las fracciones 1/10/50/100 % son prefijos anidados por clase del entrenamiento. Cada vocabulario y TF-IDF se construye únicamente desde su fracción; nunca desde validación/test.

La referencia TF-IDF usa unigramas/bigramas, min_df=2, hasta 100,000 características y regresión logística L2 optimizada con `SGDClassifier(loss='log_loss')`. Los modelos neuronales comparten `EmbeddingBag(mean) → Linear(d,128) → ReLU → Dropout(0.2) → Linear(128,4)`, Adam 1e-3 y batch 512. La entrada se adapta a la dimensión de cada embedding. El vocabulario conserva hasta 50,000 palabras con frecuencia ≥2, más padding y UNK.

Se compara inicialización aleatoria ajustable de 100d con SGNS, gensim y GloVe, cada uno congelado y ajustable. Los OOV de embeddings preentrenados parten de cero; congelados no contribuyen contenido léxico, ajustables pueden aprender. Todos comparten los mismos documentos y orden de minibatches. Máximo 10 epochs y parada después de dos epochs sin mejorar F1 macro. Se selecciona congelado/ajustable y checkpoint con validación. El manifiesto fija decisiones y hashes **antes** de evaluar test. Cada uno de los 20 modelos seleccionados (5 referencias × 4 fracciones) se evalúa una sola vez; sus predicciones y matrices quedan guardadas.''')
code('''split=load_json(ART/'news_split.json')
train=set(split['train_indices']); val=set(split['validation_indices'])
assert not train & val
subsets=[set(split['subsets'][str(f)]) for f in [.01,.1,.5,1.0]]
assert all(subsets[i]<=subsets[i+1]<=train for i in range(3))
display(pd.DataFrame([{'fracción':f,'ejemplos':len(split['subsets'][str(f)])} for f in [.01,.1,.5,1.0]]))
display(pd.DataFrame([{'modelo':m,'split':s,**r} for m,parts in split['oov'].items() for s,r in parts.items()]))
print(inspect.getsource(NewsClassifier))''')
code('''seleccion=load_json(ART/'classification_selection.json')
display(pd.DataFrame([{'modelo':r['name'],'fracción':r['fraction'],'epoch':r['best_epoch'],'parámetros':r['total_parameters'],'entrenables':r['trainable_parameters'],'segundos':r['training_seconds'],**r['validation']} for r in seleccion['all_results']]).round(5))
display(Image(filename=str(ART/'figures'/'classification_curves.png')))
display(tablas['classifiers'].round(5))
assert all(r['test_evaluations']==1 for r in load_json(ART/'evaluations'/'classification_test.json'))''')
code('''display(Image(filename=str(ART/'figures'/'classification_vs_data.png')))
display(Image(filename=str(ART/'figures'/'confusion_matrices.png')))
comparacion=tablas['classifiers'].pivot(index='fraction',columns='model',values='f1_macro')
display(comparacion.round(5))
print('Ganancia GloVe sobre aleatorio por fracción:')
display((comparacion.glove-comparacion.random).rename('Δ F1 macro'))''')
md(r'''## 5 Conclusiones y límites

Las conclusiones siguientes se calculan desde las tablas guardadas. Se distingue selección por validación de evaluación final. Tres epochs y una sola semilla limitan la evidencia: no se atribuyen diferencias pequeñas a una superioridad general ni se inventan intervalos de confianza. GloVe usa un corpus mucho mayor; su ventaja puede reflejar datos, optimización y arquitectura. Los modelos Bag ignoran orden y negación; TF-IDF con bigramas recupera parte de ese orden local. Fine-tuning puede corregir dominio/OOV, pero con pocas etiquetas también puede sobreajustar.''')
code('''comp=tablas['comparison']
clas=tablas['classifiers']
print('SGNS seleccionado:',mejor['config']['name'],'epoch',mejor['epoch'])
print('Mayor accuracy intrínseca 3CosAdd:',comp.loc[comp.analogy_add.idxmax(),'model'])
print('Mayor Spearman SimLex:',comp.loc[comp.simlex.idxmax(),'model'])
for fraction in [.01,.1,.5,1.0]:
    rows=clas[clas.fraction==fraction]
    winner=rows.loc[rows.f1_macro.idxmax()]
    print(f"Con {fraction*100:g}% de entrenamiento, mejor F1 test observado: {winner['variant']} = {winner.f1_macro:.4f}")
print('Archivo de datos y decisiones:',ART/'classification_selection.json')
print('Entorno real:',load_json(ART/'hardware.json'))''')
code('''from src.analysis import discussion
display(Markdown(discussion()))''')
md(r'''### Entregables y reproducción

- Código y dependencias: `src/`, `scripts/`, `requirements.txt`.
- Registros por epoch: `artifacts/runs/`; tablas CSV, figuras y métricas JSON.
- Predicciones finales y matrices: `artifacts/evaluations/`.
- Informe de máximo cinco páginas y versión Word editable: `entregables/`.
- [Vectores del mejor SGNS](https://github.com/iancumes/Lab7DeepLearning/releases/download/v1.0-lab7/best_sgns.kv).

Los checkpoints por epoch se preservan en el respaldo de entrenamiento; los archivos grandes no se incorporan al historial Git. Para regenerar t-SNE se cargan los tres `.kv` y se ejecuta `src.reporting.tsne`. Las coordenadas y selección de palabras ya están publicadas. Los datos se descargan desde sus fuentes originales; los hashes identifican el corpus empleado.

### Referencias
1. Mikolov et al. (2013), *Distributed Representations of Words and Phrases and their Compositionality*, NeurIPS.
2. Pennington, Socher y Manning (2014), *GloVe: Global Vectors for Word Representation*, EMNLP. https://nlp.stanford.edu/pubs/glove.pdf
3. PyTorch 2.8, documentación de Embedding y EmbeddingBag.
4. Gensim 4.4, documentación de Word2Vec y KeyedVectors.
5. [WikiText-103](https://huggingface.co/datasets/Salesforce/wikitext), [AG News](https://huggingface.co/datasets/fancyzhx/ag_news).
6. Hill, Reichart y Korhonen (2015), *SimLex-999: Evaluating Semantic Models with Similarity Estimation*, Computational Linguistics. [Recurso original](https://fh295.github.io/simlex.html).

Repositorio de la entrega: https://github.com/iancumes/Lab7DeepLearning''')
nb.cells=cells
nb.metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.11'},'colab':{'name':'Laboratorio7_Ian_Cumes_23236.ipynb'}}
nbf.write(nb,ROOT/'Laboratorio7_Ian_Cumes_23236.ipynb')
print('Notebook creado:',len(cells),'celdas')
