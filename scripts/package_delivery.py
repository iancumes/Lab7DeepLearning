"""Empaqueta código, resultados y documentos; los vectores se enlazan a la release."""
from pathlib import Path
import sys,zipfile,subprocess
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.common import ROOT,ART,load_json,save_json,sha256

assert load_json(ART/'completion.json')['complete']
assert load_json(ART/'notebook_verification.json')['passed']
assert load_json(ART/'document_verification.json')['passed']
documents=[ROOT/'Laboratorio7_Ian_Cumes_23236.ipynb']+list((ROOT/'entregables').glob('*.pdf'))+list((ROOT/'entregables').glob('*.docx'))
save_json(ART/'delivery_manifest.json',dict(files=[dict(path=str(p.relative_to(ROOT)).replace('\\','/'),bytes=p.stat().st_size,sha256=sha256(p)) for p in documents],vectors=load_json(ART/'vector_release.json')))
sources=[p for p in ROOT.iterdir() if p.is_file() and (p.suffix in ['.md','.txt','.ipynb'] or p.name in ['.gitignore','.gitattributes'])]
for folder in ['src','scripts','tests','artifacts','entregables']:
    sources += [p for p in (ROOT/folder).rglob('*') if p.is_file() and p.suffix not in ['.zip','.pyc','.pt','.kv','.npy','.pkl'] and not {'__pycache__','ids','classifiers','normalized','remote'}&set(p.relative_to(ROOT).parts)]
destination=ROOT/'entregables/Laboratorio7_Ian_Cumes_23236_entrega.zip'
with zipfile.ZipFile(destination,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
    for path in sorted(set(sources)):archive.write(path,Path(ROOT.name)/path.relative_to(ROOT))
print(destination,'bytes',destination.stat().st_size,'sha256',sha256(destination))
