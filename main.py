from multiprocessing import set_start_method, Process, Manager
from concurrent.futures import ThreadPoolExecutor, as_completed
from supabase import Client, create_client
from fastapi import FastAPI, Form, UploadFile,File, UploadFile, Form
from supabase import create_client, Client
from textract import preprocess, extract
from models import *
from typing import *
from utils import detect_gpus
from dotenv import load_dotenv
import requests
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


def processTables(queue, readers, maxThreads):
    while True:
        while queue.empty():
            print('waiting...')
            time.sleep(1)
    
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
    supabase.table('pdf_csvs').insert(toSave).execute()

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
    return {'res': f'processing page {page} of pdf_id {pdf_id}'}

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

# process from supa bucket
def process_pdf_id(queue, bot_id, pdf_id):
    path = bot_id + '/' + pdf_id
    all_imgs = supabase.storage().from_('imgs').list(path, {'limit':10000})
    all_imgs = [img['name'] for img in all_imgs]
    for i in range(len(all_imgs)):
        img = all_imgs[i]
        print(f'processing img {i} out of {len(all_imgs)}')
        f = os.path.join(bot_id, pdf_id, img)
        s = supabase.storage().from_('imgs').download(f)
        pdfJPG = PdfJPG(s, pdf_id, int(os.path.splitext(img)[0]))
        tables = preprocess(pdfJPG)
        print(f'putting {len(tables)} tables into queue...')
        #for table in tables:
        #    queue.put(table)

if __name__ == '__main__':
    set_start_method('spawn')
    global queue, gpus, readers, maxThreads
    maxThreads = 16
    queue = Manager().Queue()
    gpus = detect_gpus() 
    readers = spawn_readers(len(gpus))
    p = Process(target=processTables, args=(queue, readers, maxThreads))
    p.start()

    mode = 'single' 
    if mode == 'test':
        gpu_load_test(queue)
    elif mode == 'deploy':
        uvicorn.run(app,host='0.0.0.0', port=3000)
    elif mode == 'single':
        process_pdf_id(
            queue,
            'ff0d7ba5-0c1b-4194-9610-4411a1db63dc',
            '634eee84-ee2e-4f58-af0b-eebbd9fdc2d0'
        )

    p.join()

    
    # spawn a process that puts segs into Queue


