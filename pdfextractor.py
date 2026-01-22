from pypdf import PdfReader

def text_extractor(file_path):
    pdf_file = PdfReader(file_path)
    content = ''
    for page in pdf_file.pages:
        content += page.extract_text() + '\n'

    return content