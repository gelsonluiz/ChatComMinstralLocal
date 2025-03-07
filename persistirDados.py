import sqlite3

# Criar conexão com o banco
conn = sqlite3.connect("historico_conversas.db")
cursor = conn.cursor()

# Criar tabela para armazenar histórico
cursor.execute('''
CREATE TABLE IF NOT EXISTS conversas (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    entrada TEXT,
    resposta TEXT
)
''')
conn.commit()

# Função para salvar uma conversa
def salvar_historico(entrada, resposta):
    cursor.execute("INSERT INTO conversas (entrada, resposta) VALUES (?, ?)", (entrada, resposta))
    conn.commit()

# Função para recuperar o histórico
def recuperar_historico():
    cursor.execute("SELECT entrada, resposta FROM conversas")
    return cursor.fetchall()
