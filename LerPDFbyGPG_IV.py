import time
import persistirDados
from langchain_ollama import OllamaLLM
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

def print_with_time(message):
    print(f"{message} - {time.strftime('%Y-%m-%d %H:%M:%S')}")

def read_pdf(pdf_path):
    """ Lê um PDF e divide o texto em partes menores """
    print_with_time("Carregar o PDF")
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    print_with_time("Concatenar o texto do PDF")
    text = "\n\n".join([page.page_content for page in pages])

    print_with_time("Dividir o texto em partes menores")
    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    chunks = splitter.split_text(text)

    return chunks

# Inicializar o modelo Mistral via Ollama
print_with_time("Inicializar o modelo Mistral via Ollama")
llm = OllamaLLM(model="mistral")

# Carregar o histórico
print_with_time("Carregar o histórico")
historico = persistirDados.recuperar_historico()
historico_texto = "\n".join([f"Usuário: {h[0]}\nIA: {h[1]}" for h in historico])

# Ler e processar o PDF
pdf_path = "RESOLUÇÃO CONTRAN Nº 602.pdf"  # Defina o caminho correto do arquivo
chunks = read_pdf(pdf_path)

# Passar cada parte do texto para o modelo
print_with_time("Treinando o modelo com o texto do PDF")
for i, chunk in enumerate(chunks):
    prompt = f"{historico_texto}\nUsuário: Aprenda com este texto: {chunk}\nIA:"
    resposta = llm.invoke(prompt)
    persistirDados.salvar_historico(chunk, resposta)
    print(f"Treinado com parte {i+1}/{len(chunks)}")

# Fazer uma pergunta baseada no texto
print_with_time("Respondendo à primeira pergunta")
pergunta = "O que é GPNVE?"
prompt = f"{historico_texto}\nUsuário: {pergunta}\nIA:"
resposta = llm.invoke(prompt)
persistirDados.salvar_historico(pergunta, resposta)
print(resposta)
