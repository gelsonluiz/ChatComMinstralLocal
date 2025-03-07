import PyPDF2
from ollama import Ollama
from minstral import Minstral

def read_pdf(file_path):
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfFileReader(file)
        text = ''
        for page_num in range(reader.numPages):
            page = reader.getPage(page_num)
            text += page.extract_text()
    return text

def process_with_ollama(text):
    ollama = Ollama(api_key='your_ollama_api_key')
    result = ollama.analyze_text(text)
    return result

def process_with_minstral(text):
    minstral = Minstral(api_key='your_minstral_api_key')
    result = minstral.summarize_text(text)
    return result

def main():
    pdf_path = 'Art. 118 - Infração de trânsito cometida por veículo estrangeiro.pdf'
    pdf_text = read_pdf(pdf_path)
    
    ollama_result = process_with_ollama(pdf_text)
    print("Ollama Result:", ollama_result)
    
    minstral_result = process_with_minstral(pdf_text)
    print("Minstral Result:", minstral_result)

if __name__ == "__main__":
    main()