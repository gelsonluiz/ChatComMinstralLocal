from langchain.llms import Ollama
from langchain.document_loaders import TextLoader

# Carregar documento
loader = TextLoader("Art. 118 - Infração de trânsito cometida por veículo estrangeiro.pdf")
document = loader.load()

# Criar modelo Mistral no Ollama
llm = Ollama(model="mistral")

# Fazer pergunta sobre o documento
resumo = llm(f"Resuma esse documento: {document[0].page_content}")
print(resumo)
