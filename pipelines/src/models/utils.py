from .doc import Doc

import fitz

from pathlib import Path


def load_doc(filename, model, pages=None):
    """..."""
    filepath = Path() / 'documents'
    pdfpath = filepath / filename
    pdf = fitz.open(pdfpath)
    doc = Doc(model)
    for idx, page in enumerate(pdf):
        if pages == None:
            text_blocks = page.get_text('blocks')
            doc.add_page(idx, text_blocks)
        else:
            if idx <= pages:
                text_blocks = page.get_text('blocks')
                doc.add_page(idx, text_blocks)
    return doc