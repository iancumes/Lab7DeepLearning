# Laboratorio 7: NLP y embeddings

**Ian Cumes — carné 23236**

SGNS propio en PyTorch, comparación con gensim y `glove-wiki-gigaword-100`, y clasificación de AG News. El enunciado original está en [Laboratorio_7_NLP_Embeddings.md](Laboratorio_7_NLP_Embeddings.md).

## Ejecución reproducible

Python 3.11 o 3.12. En Colab, seleccionar GPU T4 y un runtime con Python 3.12. No se requiere un servicio pagado.

```bash
git clone https://github.com/iancumes/Lab7DeepLearning.git
cd Lab7DeepLearning
pip install -r requirements.txt
python -m pytest -q
python -u scripts/run_experiments.py
```

En CPU puede instalarse PyTorch desde su índice oficial CPU. Los datos se descargan de Hugging Face y gensim; se necesitan varios GB de espacio libre. El proceso guarda cada epoch y reanuda los experimentos terminados. `--stage prepare` prepara los datos; `--stage embeddings` completa los embeddings.

## Diseño

- WikiText-103 raw: artículos completos, semilla 23236, al menos 20 millones de tokens, subconjuntos anidados 25/50/100 %, tokenización propia y comparación con NLTK.
- Siete configuraciones SGNS de tres epochs, unigram^0.75, SGD 0.025→0.0001, `min_count=5`, subsampling `1e-4`; pares por bloques y dos tablas de embeddings.
- Selección por WordSim-353; analogías como desempate. SimLex-999 reservado para los modelos finales.
- 3CosAdd/3CosMul con vocabulario compartido de 30,000 palabras, cobertura y 14 categorías; analogías personales y paralelismo entre diferencias de vectores.
- AG News: división estratificada 90/10, fracciones anidadas 1/10/50/100 %, TF-IDF con regresión logística y `EmbeddingBag(mean) → MLP`. Embeddings aleatorios, SGNS, gensim y GloVe, congelados/ajustables seleccionados con validación. Test oficial una sola vez por modelo seleccionado.

`artifacts/` guarda métricas y manifiestos; `checkpoints/` guarda modelos locales y no se versiona. Cada vocabulario de clasificación se aprende exclusivamente de su fracción de entrenamiento. La selección se fija antes de evaluar test.

## Estado de entrega

El código está implementado; los experimentos y el informe se están verificando. Esta sección se actualizará con resultados observados y enlaces a los entregables. No se incluyen métricas simuladas.

## Referencias

- [WikiText](https://huggingface.co/datasets/Salesforce/wikitext)
- [AG News](https://huggingface.co/datasets/fancyzhx/ag_news)
- [Word2Vec en gensim](https://radimrehurek.com/gensim/models/word2vec.html)
- [GloVe, artículo original](https://nlp.stanford.edu/pubs/glove.pdf)

El tiempo publicado de GloVe para 300 dimensiones se distingue de los tiempos medidos en este laboratorio y del modelo de 100 dimensiones utilizado.
