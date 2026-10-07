# Laboratorio 7: NLP y embeddings

**Ian Cumes · carné 23236 · entrega individual**

Laboratorio completo con SGNS propio en PyTorch, Word2Vec de gensim, `glove-wiki-gigaword-100` y clasificación AG News. Se ejecutaron siete configuraciones SGNS × tres epochs, 32 variantes de clasificación y una evaluación final para cada uno de los 20 modelos seleccionados con validación. Los resultados son observados; no se simularon métricas.

## Entregables

- [Notebook ejecutado y comentado en español](Laboratorio7_Ian_Cumes_23236.ipynb)
- [Abrir notebook en Colab](https://colab.research.google.com/github/iancumes/Lab7DeepLearning/blob/main/Laboratorio7_Ian_Cumes_23236.ipynb)
- [Informe PDF, cinco páginas](entregables/Laboratorio7_Ian_Cumes_23236.pdf)
- [Informe Word editable, mismo contenido](entregables/Laboratorio7_Ian_Cumes_23236.docx)
- [Vectores del mejor SGNS `.kv`](https://github.com/iancumes/Lab7DeepLearning/releases/download/v1.0-lab7/best_sgns.kv)
- [Métricas y manifiestos](artifacts/), [figuras](artifacts/figures/), [predicciones y matrices](artifacts/evaluations/)
- [Enunciado original](Laboratorio_7_NLP_Embeddings.md)

El Word conserva texto y tablas nativos. Las tablas extensas, 54 analogías individuales, 14 categorías por método/modelo, vecinos por epoch y curvas se encuentran en el notebook.

## Resultados observados

WikiText: **20,000,727 tokens normalizados**, 5,886 artículos completos, semilla 23236. Los corpus 25/50/100% son prefijos anidados con fronteras de artículos. El corpus y vocabulario de gensim son los del SGNS seleccionado.

La selección por WordSim-353 eligió **base100, epoch 3**, dimensión 100, ventana 5, 5 negativos; ρ=0.64609185. Se usaron `min_count=5`, subsampling `1e-4`, minibatch 4096, distribución de ruido unigram^0.75 y SGD lineal 0.025→0.0001. SimLex-999 se reservó para evaluación final. Hardware de SGNS: **Tesla T4**, PyTorch 2.8.0+cu126.

F1 macro en el test oficial de AG News. Cada referencia usa la variante y checkpoint elegidos con validación para esa fracción.

| Datos de entrenamiento | TF-IDF | Aleatorio | SGNS | Gensim | GloVe |
| --- | --- | --- | --- | --- | --- |
| 1% | 81.94% | 31.20% | 71.89% | 80.68% | 66.73% |
| 10% | 89.28% | 88.75% | 89.19% | 89.03% | 89.23% |
| 50% | 91.36% | 91.62% | 91.53% | 91.58% | 91.52% |
| 100% | 91.61% | 92.10% | 92.10% | 92.03% | 92.08% |

Las tablas completas están en [epochs.csv](artifacts/epochs.csv), [classifiers.csv](artifacts/classifiers.csv), [comparison.csv](artifacts/comparison.csv) y [categories.csv](artifacts/categories.csv). El benchmark final usa exactamente 30000 candidatos compartidos y las mismas preguntas cubiertas para los tres modelos; se reporta cobertura junto con accuracy.

## Reproducción

Python 3.11/3.12. En Colab gratuito se utilizó runtime **2025.10, Python 3.12 y GPU T4**. Elegir ese runtime para las versiones fijadas; reiniciar el kernel si Colab lo solicita al instalar. No se contrató un servicio adicional. El código selecciona CPU automáticamente cuando no existe CUDA.

```bash
git clone https://github.com/iancumes/Lab7DeepLearning.git
cd Lab7DeepLearning
pip install -r requirements.txt
python -m pytest -q
```

Abrir el notebook y ejecutar todas las celdas. Por defecto reconstruye tablas/figuras desde métricas guardadas; su función `analogia(a,b,c,k)` descarga los mejores vectores de la release cuando faltan. Para repetir el entrenamiento completo, cambiar `REENTRENAR=True` o ejecutar:

```bash
python -u scripts/run_experiments.py
python scripts/verify_results.py
python scripts/build_notebook.py
python scripts/execute_notebook.py
```

`--stage prepare` prepara corpus e IDs; `--stage embeddings` completa embeddings. La ejecución reanuda epochs completas cuando están sus checkpoints. Si encuentra métricas publicadas sin pesos correspondientes, las preserva en `tmp/` e inicia un entrenamiento nuevo. `--fresh` preserva el entrenamiento anterior y fuerza otro. Nunca se seleccionan variantes con test; el manifiesto se escribe antes de evaluarlo y los resultados finales se reutilizan sin volver a calcular test.

Para CPU, instalar PyTorch 2.8 desde su índice oficial CPU antes de las demás dependencias. `requirements-lock-cpu.txt` registra el entorno Windows local; `requirements.txt` es la lista portable y `artifacts/environment.json` contiene el entorno observado de Colab. Se necesitan varios GB libres para datos, GloVe, IDs y checkpoints.

`scripts/build_reports.py` genera PDF y Word desde una única fuente de contenido; requiere las versiones de `requirements-documents.txt` y una fuente Arial, Liberation Sans o DejaVu Sans. Los documentos entregados ya están generados y revisados. `scripts/execute_notebook.py` crea un kernel temporal aislado y conserva evidencia de ejecución.

## Persistencia y verificación

Cada epoch SGNS guarda ambas tablas y el optimizador en `checkpoints/*.pt`; gensim guarda sus vectores por epoch. Los clasificadores conservan el mejor checkpoint por validación. Los respaldos se recuperaron localmente antes de cerrar Colab; archivos grandes y datasets se excluyen de Git. La release contiene un `.kv` autocontenido, sin archivos auxiliares, y respaldos ZIP de checkpoints. Para recuperar un respaldo, verificar su SHA256 y extraerlo en la raíz del repositorio; `python scripts/restore_vectors.py` reconstruye los vectores por epoch desde las tablas de entrada.

SHA256 de `best_sgns.kv`: `9da17e9a7da6d79c7e42557482979c4c3e479883e425a5e74901df60b2e7aa82`.

La auditoría [verification.json](artifacts/verification.json) comprueba 21 checkpoints, corpus mínimo, igualdad de vectores guardados, particiones anidadas y sin solapamiento de índices, hashes de selección, métricas y predicciones. [notebook_verification.json](artifacts/notebook_verification.json) registra ejecución desde kernel limpio y ausencia de errores. [document_verification.json](artifacts/document_verification.json) verifica cinco páginas en ambos formatos, nueve tablas nativas en Word y 340 fragmentos coincidentes con las métricas. Los hashes de los respaldos están en [checkpoint_backups.json](artifacts/checkpoint_backups.json); tres ZIP están en la release y los siete se conservaron localmente.

## Alcance y límites

Solo tres epochs y una semilla; no se estiman intervalos de confianza ni significancia. Gensim usa cuatro workers y actualizaciones secuenciales asíncronas; SGNS usa minibatches de 4096 pares y redibuja colisiones de negativos. Las operaciones dispersas CUDA y el paralelismo de gensim pueden introducir variación numérica al repetir, aun fijando semillas. Los pesos publicados se identifican por hash. El promedio de embeddings ignora orden y negación. GloVe usa 6 mil millones de tokens: la comparación no aísla causalmente corpus, método y optimización.

El artículo de GloVe informa dos Intel Xeon E5-2658, 85 minutos de coocurrencias y 14 minutos por iteración con **300 dimensiones y 32 cores**. Esos tiempos no se presentan como mediciones del modelo de **100 dimensiones** utilizado aquí.

## Fuentes

- [WikiText-103](https://huggingface.co/datasets/Salesforce/wikitext) y [AG News](https://huggingface.co/datasets/fancyzhx/ag_news), revisiones fijadas en código.
- [Mikolov et al. 2013](https://papers.nips.cc/paper_files/paper/2013/hash/9aa42b31882ec039965f3c4923ce901b-Abstract.html).
- [GloVe, artículo original](https://nlp.stanford.edu/pubs/glove.pdf).
- [PyTorch 2.8](https://docs.pytorch.org/docs/2.8/generated/torch.nn.EmbeddingBag.html) y [gensim Word2Vec](https://radimrehurek.com/gensim/models/word2vec.html).
- [SimLex-999](https://fh295.github.io/simlex.html); benchmarks distribuidos con gensim.
