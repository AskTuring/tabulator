import PyPDF2
from pdf2image import convert_from_path
import os

def save_first_10_pages(input_file_path, output_file_path, start, stop):
    with open(input_file_path, 'rb') as input_file, \
            open(output_file_path, 'wb') as output_file:
        pdf_reader = PyPDF2.PdfReader(input_file)
        pdf_writer = PyPDF2.PdfWriter()

        for i in range(len(pdf_reader.pages)):
            if i <= stop and i >= start:
                pdf_writer.add_page(pdf_reader.pages[i])
        pdf_writer.write(output_file)

def save_as_jpgs(ipath, opath, start, stop):
    opath += f'[{start}:{stop}]'
    images = convert_from_path(ipath, first_page=start, last_page=stop)
    if not os.path.isdir(opath): os.mkdir(opath)
    for i,img in enumerate(images):
        f = os.path.join(opath, str(i+1)+'.jpg')
        img.save(f)

def save_as_txt(ipath, opath, start, stop):
    opath += f'[{start}:{stop}]'
    with open(ipath, 'rb') as input_file, open(opath, 'a') as output_file:
        pdf_reader = PyPDF2.PdfReader(input_file)

        for i in range(len(pdf_reader.pages)):
            if i <= stop and i >= start:
                output_file.write('-'*20 + f'page {i}' + '-'*20 + '\n')
                output_file.write(pdf_reader.pages[i].extract_text())

input_file_path = '/Users/minjunes/Downloads/Siemens Manuals (chatGPT)/A/simatic-st70-complete-english-2022.pdf'
output_dir_path = os.path.join(os.getcwd(), input_file_path.split('/')[-1])
#save_first_10_page(input_file_path, output_file_path, 351,353)
#save_as_jpgs(input_file_path, output_dir_path, 1032, 1036)
save_as_txt(input_file_path, output_dir_path, 30, 30)