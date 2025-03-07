import time
from langchain_ollama import OllamaLLM
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader


def print_with_time(message):
    print(f"{message} - {time.strftime('%Y-%m-%d %H:%M:%S')}")
# Carregar o PDF
print_with_time("Carregar o PDF:\n")
pdf_path = "RESOLUÇÃO CONTRAN Nº 602.pdf"  # Substitua pelo caminho do seu arquivo
loader = PyPDFLoader(pdf_path)
pages = loader.load()

# Concatenar o texto do PDF
print_with_time("Concatenar o texto do PDF\n")
text = "\n\n".join([page.page_content for page in pages])

# Dividir o texto em partes menores (se necessário)
print_with_time("Dividir o texto em partes menores\n")
splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=500)
chunks = splitter.split_text(text)

# Inicializar o modelo Mistral no Ollama
print_with_time("Inicializar o modelo Mistral no Ollama\n")   
llm = OllamaLLM(model="mistral")

# Gerar resumo para cada parte e concatenar os resultados
print_with_time("Gerar resumo para cada parte e concatenar os resultados\n")
resumo = []
for chunk in chunks:
    response = llm.invoke(f"Resuma o seguinte texto: {chunk}")
    resumo.append(response)

# Exibir o resumo final
print_with_time("Exibir o resumo final\n")
resumo_final = "\n\n".join(resumo)
print_with_time("Resumo do PDF:\n", resumo_final)
