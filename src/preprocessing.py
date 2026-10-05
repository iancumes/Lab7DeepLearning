"""Texto → tokens → IDs, con un único tokenizador para WikiText y AG News."""
from __future__ import annotations
import collections
import os
import shutil
import re
import time
from pathlib import Path
import numpy as np
from datasets import load_dataset
from nltk.tokenize import TreebankWordTokenizer
from .common import ART, DATA, SEED, save_json, load_json, sha256

WORD = re.compile(r"[^\W\d_]+(?:'[^\W\d_]+)*|\d+(?:[.,]\d+)*|[^\w\s]", re.UNICODE)
ARTICLE = re.compile(r"^\s*=\s+([^=]+?)\s+=\s*$")
SPECIAL = re.compile(r"<unk>|<pad>", re.IGNORECASE)

def normalize(text):
    """Minúsculas, comillas ASCII y reparación de marcadores de WikiText.

    Se conservan números y puntuación como tokens: GloVe es uncased pero
    contiene ambos. No se eliminan stopwords ni se hace stemming/lemmatization.
    Los encabezados no aportan signos '=' y los especiales no son palabras.
    """
    text = text.lower().replace("’", "'").replace("‘", "'")
    text = text.replace("“", '"').replace("”", '"')
    for marker, replacement in [("@-@", "-"), ("@,@", ","), ("@.@", ".")]:
        text = text.replace(marker, replacement)
    text = SPECIAL.sub(" ", text)
    if re.match(r"^\s*=+\s.*\s=+\s*$", text):
        text = text.strip().strip("=").strip()
    return text

def tokenize(text):
    return WORD.findall(normalize(text))

def tokenizer_examples():
    examples = ["I can't believe it's already October.", "She won't buy John's laptop.",
                "A state-of-the-art computer costs $1,299.50.", "Dr. Smith lives in the U.S.A.",
                "We're testing NLP-based models.", "They've run 10,000 experiments.",
                "The king's daughter isn't a queen.", 'He said, "Hello... world!"',
                "Wikipedia uses @-@ markers and @,@ commas.",
                "Email user@example.com or visit https://uvg.edu.gt.",
                "Dogs, cats; and mice (plural nouns).", "The naïve café owner scored 99.5%."]
    library = TreebankWordTokenizer()
    return [{"sentence":x, "custom":tokenize(x), "nltk":library.tokenize(normalize(x))} for x in examples]

def iter_articles(split):
    """Un artículo empieza en un encabezado de nivel uno; no en una línea."""
    current, start, index = [], 0, -1
    for line_number, row in enumerate(split):
        text = row["text"]
        if ARTICLE.match(text):
            if current and any(t.strip() for t in current):
                yield index, start, line_number, current
            index += 1
            start, current = line_number, []
        current.append(text)
    if current and any(t.strip() for t in current):
        yield index, start, len(split), current

def prepare_corpus(target=20_000_000):
    meta = ART/"corpus.json"
    if meta.exists() and (DATA/"selected.txt").exists():
        return load_json(meta)
    started = time.perf_counter()
    wiki = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", cache_dir=str(DATA/"hf"))
    # Primero se cuentan todos los splits. Tokens normalizados y tokens por
    # espacios se reportan por separado, sin copiar cifras bibliográficas.
    counters, articles, split_stats = collections.Counter(), [], {}
    for split_name, split in wiki.items():
        stats = dict(lines=len(split), nonempty_lines=0, whitespace_tokens=0,
                     normalized_tokens=0, articles=0)
        for article_id, start, end, texts in iter_articles(split):
            normalized = [tokenize(text) for text in texts]
            count = sum(map(len, normalized))
            stats["articles"] += 1
            stats["nonempty_lines"] += sum(bool(t.strip()) for t in texts)
            stats["whitespace_tokens"] += sum(len(t.split()) for t in texts)
            stats["normalized_tokens"] += count
            if split_name == "train":
                for tokens in normalized:
                    counters.update(tokens)
                articles.append(dict(article_id=article_id, start=start, end=end, tokens=count))
        split_stats[split_name] = stats
        print("CORPUS_SPLIT", split_name, stats, flush=True)
    rng = np.random.default_rng(SEED)
    selected, total = [], 0
    for i in rng.permutation(len(articles)):
        selected.append(articles[int(i)])
        total += articles[int(i)]["tokens"]
        if total >= target:
            break
    assert total >= target
    # El orden aleatorio se conserva; subconjuntos anidados por artículos.
    selected_counts = collections.Counter()
    boundaries, tokens_so_far, lines_so_far = [], 0, 0
    with (DATA/"selected.txt").open("w", encoding="utf-8", newline="\n") as out:
        for item in selected:
            for row in wiki["train"].select(range(item["start"], item["end"])):
                tokens = tokenize(row["text"])
                if not tokens:
                    continue
                selected_counts.update(tokens)
                tokens_so_far += len(tokens)
                # Límite para evitar el truncado de frases largas en gensim.
                for pos in range(0, len(tokens), 1000):
                    out.write(" ".join(tokens[pos:pos+1000]) + "\n")
                    lines_so_far += 1
            boundaries.append(dict(tokens=tokens_so_far, sentences=lines_so_far,
                                   article_id=item["article_id"]))
    assert tokens_so_far == total
    manifest = dict(seed=SEED, target_tokens=target, selected_tokens=total,
                    selected_articles=len(selected), selected_sentences=lines_so_far,
                    articles=selected, article_boundaries=boundaries,
                    splits=split_stats, selected_sha256=sha256(DATA/"selected.txt"),
                    preprocessing_seconds=time.perf_counter()-started,
                    normalization="lowercase; preserve punctuation/numbers; repair @ markers; remove header '=' and special tokens")
    save_json(ART/"frequencies_full_train.json", counters)
    save_json(ART/"frequencies_selected.json", selected_counts)
    frequencies = np.array(sorted(counters.values(), reverse=True), dtype=np.int64)
    manifest["zipf_coverage"] = {str(k):float(frequencies[:k].sum()/frequencies.sum()) for k in [10,1000,30000]}
    manifest["thresholds"] = [{"min_count":m, "vocabulary_size":sum(v>=m for v in selected_counts.values()),
                               "unknown_token_fraction":sum(v for v in selected_counts.values() if v<m)/total}
                              for m in [1,5,10]]
    save_json(ART/"tokenizer_examples.json", tokenizer_examples())
    save_json(meta, manifest)
    return manifest

def prepare_ids(config, manifest):
    folder = ART/"ids"/config["name"]
    folder.mkdir(parents=True, exist_ok=True)
    if (folder/"meta.json").exists():
        cached=load_json(folder/"meta.json")
        for key in ["fraction","min_count","sample","window"]:
            assert cached["config"][key]==config[key],f"Cache incompatible: {config['name']} {key}"
        if cached["config"]!=config:
            cached["config"]=config
            save_json(folder/"meta.json",cached)
        return cached
    base = ART/"ids"/"base100"
    if config["name"]!="base100" and config["fraction"]==1.0 and (base/"meta.json").exists():
        base_meta=load_json(base/"meta.json")
        if base_meta["config"]["min_count"]==config["min_count"] and base_meta["config"]["sample"]==config["sample"]:
            for filename in ["ids.npy","segments.npy","counts.npy","keep.npy","corpus.txt"]:
                destination=folder/filename
                if not destination.exists():
                    try:
                        os.link(base/filename,destination)
                    except OSError:
                        shutil.copyfile(base/filename,destination)
            base_meta["config"]=config
            base_meta["pairs_before_subsampling"]=pair_count(np.load(folder/"segments.npy",mmap_mode="r"),config["window"])
            save_json(folder/"meta.json",base_meta)
            return base_meta
    boundary = next(b for b in manifest["article_boundaries"]
                    if b["tokens"] >= manifest["selected_tokens"]*config["fraction"])
    counts = collections.Counter()
    # Una primera pasada determina el vocabulario del corpus de ESTA iteración.
    with (DATA/"selected.txt").open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= boundary["sentences"]:
                break
            counts.update(line.split())
    vocabulary = sorted((w for w,c in counts.items() if c>=config["min_count"]), key=lambda w:(-counts[w],w))
    mapping = {w:i for i,w in enumerate(vocabulary)}
    ids, segments = [], []
    corpus_path = folder/"corpus.txt"
    with (DATA/"selected.txt").open(encoding="utf-8") as f, corpus_path.open("w", encoding="utf-8") as out:
        for i, line in enumerate(f):
            if i >= boundary["sentences"]:
                break
            # Gensim y SGNS reciben exactamente los mismos tokens elegibles.
            words = [w for w in line.split() if w in mapping]
            if not words:
                continue
            out.write(" ".join(words)+"\n")
            ids.extend(mapping[w] for w in words)
            segments.extend([i]*len(words))
    np.save(folder/"ids.npy", np.asarray(ids, dtype=np.int32))
    np.save(folder/"segments.npy", np.asarray(segments, dtype=np.int32))
    frequencies = np.array([counts[w] for w in vocabulary], dtype=np.float64)
    np.save(folder/"counts.npy", frequencies)
    # Fórmula original de Word2Vec; frecuencias sobre tokens elegibles.
    f = frequencies / frequencies.sum()
    keep = np.minimum(1.0, (np.sqrt(f/config["sample"])+1)*(config["sample"]/f))
    np.save(folder/"keep.npy", keep.astype(np.float32))
    special_stats = {w:{"occurrences":counts[w], "expected_discard_fraction":float(1-keep[mapping[w]])}
                     for w in ["the","of","and"] if w in mapping}
    pairs = pair_count(np.asarray(segments, dtype=np.int32), config["window"])
    meta = dict(config=config, vocabulary=vocabulary, corpus_tokens=boundary["tokens"],
                eligible_tokens=len(ids), vocabulary_size=len(vocabulary),
                unknown_fraction=(boundary["tokens"]-len(ids))/boundary["tokens"],
                pairs_before_subsampling=pairs, corpus_sha256=sha256(corpus_path),
                subsampling=special_stats, sentences=boundary["sentences"])
    save_json(folder/"meta.json", meta)
    return meta

def pair_count(segments, window):
    return int(sum(2*np.count_nonzero(segments[distance:]==segments[:-distance])
                   for distance in range(1,min(window,len(segments)-1)+1)))

def pair_batches(ids, segments, window, batch_size, rng, block_size=65536):
    """Pares ordenados central-contexto; nunca cruzan una frontera de frase."""
    for start in rng.permutation(np.arange(0,len(ids),block_size)):
        pos = np.arange(start,min(start+block_size,len(ids)))
        centers, contexts = [], []
        for distance in range(-window,window+1):
            if distance == 0:
                continue
            neighbor = pos + distance
            mask = (neighbor>=0)&(neighbor<len(ids))
            p, q = pos[mask], neighbor[mask]
            inside = segments[p]==segments[q]
            centers.append(ids[p[inside]])
            contexts.append(ids[q[inside]])
        if not centers:
            continue
        c, o = np.concatenate(centers), np.concatenate(contexts)
        order = rng.permutation(len(c))
        for begin in range(0,len(order),batch_size):
            indices = order[begin:begin+batch_size]
            yield c[indices], o[indices]
