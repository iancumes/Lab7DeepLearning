"""Prepara respaldos atómicos durante Colab para descarga HTTP privada.

No sube datos a un servicio externo: escribe ZIP en el directorio indicado.
El usuario descarga los enlaces servidos por el proxy autenticado de Colab.
"""
from pathlib import Path
import argparse,json,zipfile,time,hashlib
ROOT=Path(__file__).resolve().parents[1]

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def archive(destination,sources):
    temporary=destination.with_suffix('.building')
    with zipfile.ZipFile(temporary,'w',zipfile.ZIP_STORED) as z:
        for path in sources:z.write(path,path.relative_to(ROOT))
    temporary.replace(destination)
    meta=dict(file=destination.name,bytes=destination.stat().st_size,sha256=digest(destination))
    destination.with_suffix('.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print('RESPALDO_LISTO',meta,flush=True)

def main(output):
    output.mkdir(parents=True,exist_ok=True)
    while True:
        for path in (ROOT/'artifacts/runs').glob('*.json'):
            if path.stem=='gensim':continue
            try:rows=json.loads(path.read_text(encoding='utf-8'))
            except (ValueError,FileNotFoundError):continue
            if len(rows)!=3:continue
            destination=output/f'lab7_{path.stem}_checkpoints.zip'
            if not destination.exists():
                sources=sorted((ROOT/'checkpoints').glob(path.stem+'_e*.pt'))
                if len(sources)==3:archive(destination,sources+[path])
        completion=ROOT/'artifacts/completion.json'
        if completion.exists():
            vector=ROOT/'checkpoints/best_sgns.kv'
            (ROOT/'artifacts/vector_release.json').write_text(json.dumps(dict(file=vector.name,bytes=vector.stat().st_size,sha256=digest(vector),release='v1.0-lab7',url='https://github.com/iancumes/Lab7DeepLearning/releases/download/v1.0-lab7/best_sgns.kv'),indent=2),encoding='utf-8')
            sources=[p for p in (ROOT/'artifacts').rglob('*') if p.is_file() and not {'ids','classifiers','normalized','remote'}&set(p.relative_to(ROOT/'artifacts').parts) and not p.name.endswith('.tmp')]
            sources+=[ROOT/'checkpoints/best_sgns.kv']
            archive(output/'lab7_resultados_finales.zip',sources)
            archive(output/'lab7_clasificadores.zip',[p for p in (ROOT/'artifacts/classifiers').rglob('*') if p.is_file()])
            archive(output/'lab7_gensim.zip',[p for p in (ROOT/'checkpoints').glob('gensim*.kv*') if p.is_file()])
            break
        time.sleep(10)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=Path('/content/exports'))
    main(parser.parse_args().output)
