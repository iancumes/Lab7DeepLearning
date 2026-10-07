"""Comprueba contenido nativo y cifras; ejecutar después de revisar cada PNG.

Usa el runtime de documentos y exige confirmación explícita de la revisión
visual realizada, además de las comprobaciones automáticas de contenido.
"""
from pathlib import Path
import argparse
import hashlib
import json
import re
import sys
import unicodedata
from docx import Document
from pypdf import PdfReader

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.build_reports import build_content,URL

def normalized(text):
    return re.sub(r'\s+','',unicodedata.normalize('NFKC',str(text)))

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main(visual_review_complete):
    assert visual_review_complete,'Revisar primero todas las páginas renderizadas de ambos documentos.'
    output=ROOT/'entregables'
    stem='Laboratorio7_Ian_Cumes_23236'
    content=json.loads((output/'contenido_informe.json').read_text(encoding='utf-8'))
    assert content==json.loads(json.dumps(build_content(),ensure_ascii=False)), 'Las métricas guardadas cambiaron después de generar el informe.'
    docx_path=output/(stem+'.docx'); pdf_path=output/(stem+'.pdf')
    doc=Document(docx_path); pdf=PdfReader(pdf_path)
    rendered_word=ROOT/'tmp/word_render'/(stem+'.pdf')
    word_pdf=PdfReader(rendered_word)
    assert len(pdf.pages)==len(word_pdf.pages)==len(content)==5
    native_word=normalized(''.join(doc.element.itertext()))
    tables_expected=sum(block[0]=='table' for page in content for block in page)
    assert len(doc.tables)==tables_expected and len(doc.inline_shapes)==0
    fragments_checked=0
    for index,blocks in enumerate(content):
        pdf_text=normalized(pdf.pages[index].extract_text())
        word_text=normalized(word_pdf.pages[index].extract_text())
        for block in blocks:
            fragments=[block[1]] if block[0] in ['h','p'] else [str(value) for row in [block[1]]+block[2] for value in row]
            for fragment in fragments:
                expected=normalized(fragment)
                assert expected in native_word,('Word nativo',index+1,fragment)
                assert expected in pdf_text,('PDF',index+1,fragment)
                assert expected in word_text,('Word renderizado',index+1,fragment)
                fragments_checked+=1
    assert normalized(pdf.pages[-1].extract_text()).endswith(normalized(URL))
    links=[str(a.get_object().get('/A',{}).get('/URI','')) for page in pdf.pages for a in page.get('/Annots',[])]
    assert URL in links
    pdf_png=list((ROOT/'tmp/pdf_render').glob('page-*.png'))
    word_png=list((ROOT/'tmp/word_render').glob('page-*.png'))
    assert len(pdf_png)==len(word_png)==5
    result=dict(passed=True,pdf_pages=5,word_pages=5,native_word_tables=tables_expected,
                native_word_text=True,matching_source_metrics=True,same_content=True,
                fragments_checked=fragments_checked,repository_link_verified=True,
                all_pdf_pages_visually_reviewed=True,all_word_pages_visually_reviewed=True,
                files=[dict(path=str(p.relative_to(ROOT)).replace('\\','/'),sha256=digest(p)) for p in [pdf_path,docx_path]])
    (ROOT/'artifacts/document_verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--visual-review-complete',action='store_true')
    main(parser.parse_args().visual_review_complete)
