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
            

input_file_path = '/Users/minjunes/Downloads/Siemens Manuals (chatGPT)/B/s71200_system_manual_en-US_en-US.pdf'
output_dir_path = os.path.join(os.getcwd(), input_file_path.split('/')[-1])
#save_first_10_page(input_file_path, output_file_path, 351,353)
save_as_jpgs(input_file_path, output_dir_path, 1032, 1036)