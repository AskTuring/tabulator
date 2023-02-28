from pydantic import BaseModel

class PdfJPG(object):
    def __init__(self, jpg, pdf_id, page):
        self.pdf_id = pdf_id
        self.jpg = jpg
        self.page = page

class PdfTable(BaseModel):
    pdf_id: str
    table_no: int
    page: int

class Table:
    def __init__(self, src, rows, cols, meta: PdfTable):
        self.src = src
        self.rows = rows
        self.cols = cols
        self.meta = meta
