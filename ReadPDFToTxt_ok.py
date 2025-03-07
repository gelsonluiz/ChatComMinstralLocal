import fitz  # PyMuPDF

def pdf_para_txt(arquivo_pdf, arquivo_txt):
    # Abre o PDF
    doc = fitz.open(arquivo_pdf)
    texto = ""

    # Percorre todas as páginas e extrai o texto
    for pagina in doc:
        texto += pagina.get_text() + "\n"

    # Salva o texto em um arquivo .txt
    with open(arquivo_txt, "w", encoding="utf-8") as f:
        f.write(texto)

    print(f"Texto salvo em {arquivo_txt}")

# Exemplo de uso
pdf_para_txt(r"Teste_1.pdf", "saida.txt")
