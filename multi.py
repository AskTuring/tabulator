from multiprocessing import Queue
from multiprocessing import Process
from pydantic import BaseModel
import logging
import os, json
import time
from itertools import cycle

class PdfJPG(BaseModel):
    pdf_id: str
    jpg: bytearray
    page: int

class PdfTable(BaseModel):
    pdf_id: str
    table_no: int
    page: int

class Table:
    def __init__(self, src, metadata: PdfTable):
        self.src = src
        self.metadata = metadata

def process_jpg(jpg: PdfJPG):
    pass

# TODO
'''
    A) 
    --process 1--
    receive HTTP request to process JPG img
    process the img immediately using CPU, then put it into queue
    queue: List[segs]

    --process 2--
    each gpu pops from queue and spits out result 
    collect X results, save to CSV
    HTTP to supabase

'''

if __name__ == '__main__':
    queue = Queue()
    
    # spawn a process that puts segs into Queue


