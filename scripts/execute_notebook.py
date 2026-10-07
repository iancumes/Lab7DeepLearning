"""Ejecuta el notebook desde un kernel nuevo, sin reentrenar ni reevaluar test."""
from pathlib import Path
import os,sys,json,time
import nbformat
from nbclient import NotebookClient

ROOT=Path(__file__).resolve().parents[1]
kernel=ROOT/'tmp/jupyter/kernels/lab7'
kernel.mkdir(parents=True,exist_ok=True)
(kernel/'kernel.json').write_text(json.dumps(dict(argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],display_name='Laboratorio 7',language='python')),encoding='utf-8')
os.environ['JUPYTER_PATH']=str(ROOT/'tmp/jupyter')+os.pathsep+os.environ.get('JUPYTER_PATH','')
path=ROOT/'Laboratorio7_Ian_Cumes_23236.ipynb'
notebook=nbformat.read(path,as_version=4)
started=time.perf_counter()
NotebookClient(notebook,timeout=600,kernel_name='lab7',allow_errors=False,resources={'metadata':{'path':str(ROOT)}}).execute()
with path.open('w',encoding='utf-8',newline='\n') as output:
    nbformat.write(notebook,output)
cells=[c for c in notebook.cells if c.cell_type=='code']
assert all(c.execution_count is not None for c in cells)
assert all(o.output_type!='error' for c in cells for o in c.get('outputs',[]))
evidence=dict(passed=True,clean_kernel=True,code_cells=len(cells),error_outputs=0,seconds=time.perf_counter()-started,python=sys.version)
(ROOT/'artifacts/notebook_verification.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2),encoding='utf-8')
print(evidence)
