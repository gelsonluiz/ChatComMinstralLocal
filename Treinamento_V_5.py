import time
import faiss
import numpy as np
import persistirDados
from langchain_ollama import OllamaLLM
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings

# Inicializar modelo de embeddings para busca semântica
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def print_with_time(message):
    print(f"{message} - {time.strftime('%Y-%m-%d %H:%M:%S')}")

# Função para ler e dividir o PDF
def read_pdf(pdf_path):
    print_with_time("Carregar o PDF")
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    print_with_time("Concatenar o texto do PDF")
    text = "\n\n".join([page.page_content for page in pages])

    print_with_time("Dividir o texto em partes menores")
    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    chunks = splitter.split_text(text)

    return chunks

# Criar índice FAISS para buscas rápidas
def create_faiss_index(chunks):
    print_with_time("Criando índice FAISS")
    vector_store = faiss.IndexFlatL2(384)  # 384 dimensões (MiniLM-L6-v2)
    vectors = [embeddings.embed_query(chunk) for chunk in chunks]
    vector_store.add(np.array(vectors, dtype=np.float32))
    return vector_store, chunks

# Função para buscar trechos mais relevantes com FAISS
def retrieve_relevant_text(query, vector_store, chunks, top_k=3):
    query_vector = np.array([embeddings.embed_query(query)], dtype=np.float32)
    _, indices = vector_store.search(query_vector, top_k)
    return "\n\n".join([chunks[i] for i in indices[0]])

# Inicializar o modelo Mistral via Ollama
print_with_time("Inicializar o modelo Mistral via Ollama")
llm = OllamaLLM(model="mistral")

# Carregar histórico
print_with_time("Carregar o histórico")
historico = persistirDados.recuperar_historico()
historico_texto = "\n".join([f"Usuário: {h[0]}\nIA: {h[1]}" for h in historico])

# Carregar e indexar o PDF
pdf_path = "RESOLUÇÃO CONTRAN Nº 602.pdf"
chunks = read_pdf(pdf_path)
vector_store, indexed_chunks = create_faiss_index(chunks)

# Criar um prompt de aprendizado baseado no histórico
entrada = "Texto de histórico de conversa para aprendizado"
prompt = f"{historico_texto}\nUsuário: {entrada}\nIA:"

# Treinar o modelo com o texto recuperado
print_with_time("Treinando o modelo com texto relevante")
texto_relevante = retrieve_relevant_text("Aprenda sobre a Resolução CONTRAN 602", vector_store, indexed_chunks)
texto = llm.invoke(f"Aprenda o seguinte texto:\n\n{texto_relevante}")
print(texto)

# Exemplo de consulta ao modelo após aprendizado
print_with_time("Realizando pergunta ao modelo")
pergunta = "O que é GPNVE?"
texto_relevante = retrieve_relevant_text(pergunta, vector_store, indexed_chunks)
resposta = llm.invoke(f"Com base no seguinte texto:\n\n{texto_relevante}\n\nResponda: {pergunta}")
print(resposta)
print_with_time("Resposta")