import requests
import base64
import os
import time

JPGDIR = '/Users/minjunes/tabulator/data/simatic-st70-complete-english-2022.pdf'
URL = 'http://127.0.0.1:3000/api/textract'

def send(f):
    print(f'sending jpg {f}')
    res = requests.post(
        URL,
        files={
            'file': open(f, 'rb'),
        },
        data={
            'pdf_id': 'mock',
            'page': 0
        }
        
    )
    print(res.text)

if __name__ == '__main__':
    # 11 pdf docs
    for i in range(10):
        for f in os.listdir(JPGDIR)[:-1]:
            f = os.path.join(JPGDIR,f)
            send(f)