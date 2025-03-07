import time
import persistirDados

from langchain_ollama import OllamaLLM
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

def print_with_time(message):
    print(f"{message} - {time.strftime('%Y-%m-%d %H:%M:%S')}")

def read_pdf(pdf_path):
    # Carregar o PDF
    print_with_time("Carregar o PDF:\n")
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    # Concatenar o texto do PDF
    print_with_time("Concatenar o texto do PDF\n")
    text = "\n\n".join([page.page_content for page in pages])

    # Dividir o texto em partes menores para o modelo processar melhor
    print_with_time("Dividir o texto em partes menores\n")
    splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=500)
    chunks = splitter.split_text(text)
    
    return chunks

# Caminho do arquivo PDF
#pdf_path = r"RESOLUÇÃO CONTRAN Nº 602.pdf"
#chunks = read_pdf(pdf_path)

# Inicializar o modelo Mistral via Ollama
print_with_time("Inicializar o modelo Mistral via Ollama\n")
llm = OllamaLLM(model="mistral")

# Resumir cada trecho do PDF
#print_with_time("Resumir cada trecho do PDF\n")
#texto = [llm.invoke(f"Aprenda com o seguinte texto: {chunk}") for chunk in chunks]
#textoUnificado = "\n\n".join(texto)
#print("Texto Unificado:\n", textoUnificado)


# Carregar o histórico
print_with_time("Carregar o histórico\n")
historico = persistirDados.recuperar_historico()
historico_texto = "\n".join([f"Usuário: {h[0]}\nIA: {h[1]}" for h in historico])
entrada = "Texto de historico de conversa para aprendizado"
prompt = f"{historico_texto}\nUsuário: {entrada}\nIA:"

#texto = llm.invoke(f"Como um Product Owner Senior quero que responda todas as perguntas abaixo conforme os textos a seren repassados")
#print(texto)

print_with_time("Primeiro Prompt\n")
texto = llm.invoke(f"Como um Product Owner Senior quero que responda todas as perguntas abaixo conforme os textos a seren repassados. Agora, aprenda com o seguinte texto: {prompt}")
print(texto)

#texto = llm.invoke(f"Agora, resuma o texto aprendido")
#print(texto)

print_with_time("Primeira pergunta\n")
texto = llm.invoke(f"E responda: o que é GPNVE?")
print(texto)

#print("Primeira Persistência")
#persistirDados.salvar_conversa('Aprenda com o seguinte texto', textoUnificado)

#resumo = [llm.invoke(f"Resuma o seguinte texto: {textoUnificado}")]

# Juntar os resumos 
#print_with_time("Juntar os resumos\n")
#print("Resumo do PDF:\n", resumo)

# Salvar no histórico
#print_with_time("Salvar no histórico\n")
#persistirDados.salvar_conversa(resumo, "Resuma o seguinte texto:")
#print_with_time('Fim do processamento\n')