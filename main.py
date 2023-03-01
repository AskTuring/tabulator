from multiprocessing import Queue, Process, Manager
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

def processTables(queue, gpus, max_gpu_load):
    while queue.empty():
        print('waiting...')
        time.sleep(1)
    
    maxChunkSize = len(gpus) * max_gpu_load
    allChunks = []
    chunk = []
    while not queue.empty():
        if len(chunk) < maxChunkSize:
            chunk.append(queue.get())
        else:
            allChunks.append(chunk)
            chunk = []
    if chunk:
        allChunks.append(chunk)


    for chunk in allChunks:         
        assert(len(chunk) <= maxChunkSize)
        threads = []
        for i, table in enumerate(chunk):
            i %= len(gpus)
            thread = threading.Thread(target=extract, args=(table, gpus[i]))
            threads.append(thread)
            thread.start()
    processTables(queue, gpus, max_gpu_load)

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
    testdir = '/Users/minjunes/tabulator/data/simatic-st70-complete-english-2022.pdf'
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
    global queue, gpus, readers, max_gpu_load
    max_gpu_load = 8
    queue = Manager().Queue()
    gpus = detect_gpus() 
    readers = spawn_readers(len(gpus))
    p = Process(target=processTables, args=(queue, readers, max_gpu_load))
    p.start()
    test = True
    if test:
        gpu_load_test(queue)
    else:
        uvicorn.run(app,host='0.0.0.0', port=3000)
    p.join()

    
    # spawn a process that puts segs into Queue


