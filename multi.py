from multiprocessing import Queue, Process, Manager
from itertools import cycle
from fastapi import FastAPI, Form, UploadFile, HTTPException, status, Request, File, UploadFile, Form
from supabase import create_client, Client
from main import preprocess, extract
from models import *
from typing import *
from utils import detect_gpus
import easyocr
import threading
import uvicorn
import logging
import os, json
import time
import random

app = FastAPI()

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

# TODO:
'''
    concern... wouldn't using GPU with 5-6 jobs overload it's memory and crash it?  
    how to "stop" when memory usage is too high? 

'''
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

@app.post('/api/textract')
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

if __name__ == '__main__':
    global queue, gpus, readers, max_gpu_load
    max_gpu_load = 4
    queue = Manager().Queue()
    gpus = detect_gpus() 
    readers = spawn_readers(len(gpus))
    p = Process(target=processTables, args=(queue, readers, max_gpu_load))
    p.start()
    uvicorn.run(app,host='0.0.0.0', port=3000)
    p.join()

    
    # spawn a process that puts segs into Queue


