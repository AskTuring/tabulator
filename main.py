from multiprocessing import set_start_method, Process, Manager
from concurrent.futures import ThreadPoolExecutor, as_completed
from supabase import Client, create_client
from itertools import cycle
from fastapi import FastAPI, Form, UploadFile, HTTPException, status, Request, File, UploadFile, Form
from supabase import create_client, Client
from textract import preprocess, extract
from models import *
from typing import *
from utils import detect_gpus
from dotenv import load_dotenv
import easyocr
import threading
import uvicorn
import logging
import os, json
import time
import random
import uuid

load_dotenv()
app = FastAPI(root_path=os.getenv('ROOT_PATH'))
supabase: Client = create_client(os.environ.get('SUPABASE_URL'), os.environ.get('SUPABASE_KEY'))

def sim_extract(table: Table, i):
    time.sleep(random.uniform(0,2))
    print(f'gpuIdx {i}: ', table.meta)

def sim(queue, gpus, max_gpu_load):
    while queue.empty():
        print('waiting...')
        time.sleep(1)
    
                # replace with len(gpus)
    num_gpus = 4
    maxChunkSize = num_gpus * max_gpu_load 
    allChunks = []
    chunk = []
    while not queue.empty():
        if len(chunk) < maxChunkSize:
            chunk.append(queue.get())
        else:
            allChunks.append(chunk)
            chunk = []
    print('len allCHunks', len(allChunks)) 
    if chunk:
        allChunks.append(chunk)
    
    for chunk in allChunks:         
        assert(len(chunk) <= maxChunkSize)
        threads = []
        s = time.time()
        for i, table in enumerate(chunk):
            gpuIdx = i % num_gpus
            thread = threading.Thread(target=sim_extract, args=(table, gpuIdx))
            threads.append(thread)
            thread.start()
            print(f'starting thread {i}...')
        for thread in threads:
            thread.join()
        e = time.time()
        print('time elapsed: ',e-s)
    sim(queue, gpus, max_gpu_load)

def processTables(queue, readers):
    while queue.empty():
        print('waiting...')
        time.sleep(1)
    
    maxThreads = 32
    allChunks = []
    chunk = []
    while not queue.empty():
        if len(chunk) < maxThreads:
            chunk.append(queue.get())
        else:
            allChunks.append(chunk)
            chunk = []
    if chunk:
        allChunks.append(chunk)

    for chunk in allChunks:         
        assert(len(chunk) <= maxThreads)
        with ThreadPoolExecutor(max_workers=maxThreads) as executor:
            futures = []
            for i, table in enumerate(chunk):
                future = executor.submit(extract, table, readers[i%len(readers)])
                futures.append(future)
            
            for future in as_completed(futures):
                try:
                    table, tableCsv = future.result()
                    save_table(table, tableCsv)
                except Exception as e:
                    print(e)

    processTables(queue, readers)

class SaveTable(BaseModel):
    pdf_id: str
    table_no: int
    page: int
    content: str

def save_table(table: Table, tableCsv: str):
    toSave: SaveTable = {
        'pdf_id': table.meta['pdf_id'],
        'table_no': table.meta['table_no'],
        'page': table.meta['page'],
        'content': tableCsv
    }
    supabase.table('pdf_tables').insert(toSave).execute()

@app.get('/')
def test():
    return {'res': 'process GPU server functional'}

@app.post('/textract')
async def textract(
    file: UploadFile=File(...),
    pdf_id:str=Form(...),
    page:int=Form(...)
    ):
    pdfJPG: PdfJPG = PdfJPG(await file.read(), pdf_id, page)
    tables: List[Table] = preprocess(pdfJPG)
    for table in tables:
        queue.put(table)
    return {'res': 'processing...'}

def spawn_readers(num_gpus):
    readers = []
    for i in range(num_gpus):
        reader = easyocr.Reader(
           ['en'],
            gpu=f'cuda:{i}'
        )
        readers.append(reader)
    return readers

def gpu_load_test(queue, iters=5):
    testdir = os.path.join(os.getcwd(),'/data/simatic-st70-complete-english-2022.pdf')
    for i in range(iters):
        fs = os.listdir(testdir)
        for j,f in enumerate(fs):
            f = os.path.join(testdir, f)
            with open(f, 'rb') as file:
                pdfJPG = PdfJPG(file.read(), str(uuid.uuid4()), len(fs)*i+j)
                tables = preprocess(pdfJPG)
                for table in tables:
                    queue.put(table)

if __name__ == '__main__':
    set_start_method('spawn')
    global queue, gpus, readers
    queue = Manager().Queue()
    gpus = detect_gpus() 
    readers = spawn_readers(len(gpus))
    p = Process(target=processTables, args=(queue, readers))
    p.start()
    test = True
    if test:
        gpu_load_test(queue)
    else:
        uvicorn.run(app,host='0.0.0.0', port=3000)
    p.join()

    
    # spawn a process that puts segs into Queue


