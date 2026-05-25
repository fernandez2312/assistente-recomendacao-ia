import chromadb
from sentence_transformers import SentenceTransformer

print("Inicializando o modelo de Embeddings (Busca Semântica)...")
# Modelo leve e otimizado para transformar textos em vetores numéricos
modelo_vetorial = SentenceTransformer("all-MiniLM-L6-v2")

def inicializar_banco_vetorial():
    # Cria ou conecta ao banco vetorial armazenado localmente na pasta 'dados_vetoriais'
    cliente_chroma = chromadb.PersistentClient(path="./dados_vetoriais")
    
    # Cria uma coleção de vetores (equivalente a uma tabela no banco tradicional)
    colecao = cliente_chroma.get_or_create_collection(name="recomendacao_filmes")
    
    # Base de dados de exemplo com descrições detalhadas
    filmes = [
        {"id": "1", "titulo": "A Origem", "descricao": "Um ladrão que rouba segredos corporativos por meio do uso de tecnologia de compartilhamento de sonhos recebe a tarefa inversa de plantar uma ideia na mente de um CEO."},
        {"id": "2", "titulo": "Interestelar", "descricao": "Uma equipe de exploradores viaja através de um buraco de minhoca no espaço em uma tentativa de garantir a sobrevivência da humanidade visitando novos planetas."},
        {"id": "3", "titulo": "O Poderoso Chefão", "descricao": "O patriarca envelhecido de uma dinastia do crime organizado transfere o controle de seu império clandestino para seu filho relutante em meio a guerras de mafiosos."},
        {"id": "4", "titulo": "Matrix", "descricao": "Um hacker de computador aprende com rebeldes misteriosos sobre a verdadeira natureza de sua realidade e seu papel na guerra contra seus controladores de inteligência artificial."},
        {"id": "5", "titulo": "Se Beber, Não Case!", "descricao": "Três amigos acordam em Las Vegas após uma despedida de solteiro alucinante, sem memória da noite anterior e percebendo que o noivo sumiu."}
    ]
    
    print("Convertendo sinopses em vetores e salvando no ChromaDB...")
    
    for filme in filmes:
        # Transforma a descrição do filme em uma lista de números (vetor)
        vetor_descricao = modelo_vetorial.encode(filme["descricao"]).tolist()
        
        # Salva o vetor e os dados de texto no banco
        colecao.add(
            ids=[filme["id"]],
            embeddings=[vetor_descricao],
            documents=[filme["descricao"]],
            metadatas=[{"titulo": filme["titulo"]}]
        )
        
    print(f"Sucesso! {len(filmes)} filmes indexados no banco vetorial.")

if __name__ == "__main__":
    inicializar_banco_vetorial()
