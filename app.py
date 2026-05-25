import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Recomendador de Filmes IA", page_icon="🎬")

st.title("🎬 Assistente de Recomendação com Busca Semântica (RAG)")
st.write("Diga o que você está com vontade de assistir e nossa IA encontrará o filme ideal por contexto!")

# 1. Base de dados na memória para evitar travas de arquivos no Windows
filmes = [
    {"id": "1", "titulo": "A Origem", "descricao": "Um ladrão que rouba segredos corporativos por meio do uso de tecnologia de compartilhamento de sonhos recebe a tarefa inversa de plantar uma ideia na mente de um CEO."},
    {"id": "2", "titulo": "Interestelar", "descricao": "Uma equipe de exploradores viaja através de um buraco de minhoca no espaço em uma tentativa de garantir a sobrevivência da humanidade visitando novos planetas."},
    {"id": "3", "titulo": "O Poderoso Chefão", "descricao": "O patriarca envelhecido de uma dinastia do crime organizado transfere o controle de seu império clandestino para seu filho relutante em meio a guerras de mafiosos."},
    {"id": "4", "titulo": "Matrix", "descricao": "Um hacker de computador aprende com rebeldes misteriosos sobre a verdadeira natureza de sua realidade e seu papel na guerra contra seus controladores de inteligência artificial."},
    {"id": "5", "titulo": "Se Beber, Não Case!", "descricao": "Três amigos acordam em Las Vegas após uma despedida de solteiro alucinante, sem memória da noite anterior e percebendo que o noivo sumiu."}
]

# Converter para DataFrame do Pandas para facilitar a manipulação
df_filmes = pd.DataFrame(filmes)

@st.cache_resource
def carregar_modelo_e_vetores():
    modelo = SentenceTransformer("all-MiniLM-L6-v2")
    # Gera os embeddings (vetores) de todos os filmes da nossa base
    vetores_filmes = modelo.encode(df_filmes['descricao'].tolist())
    return modelo, vetores_filmes

modelo_vetorial, vetores_filmes = carregar_recursos = carregar_modelo_e_vetores()

# Campo de entrada do usuário
entrada_usuario = st.text_input("O que você quer assistir hoje?", placeholder="Ex: Um filme tenso no espaço ou uma comédia louca com amigos")

if entrada_usuario:
    with st.spinner("Buscando e calculando proximidade semântica..."):
        # 2. Transforma o texto inserido pelo usuário em vetor
        vetor_pergunta = modelo_vetorial.encode([entrada_usuario])
        
        # 3. Calcula a similaridade de cosseno entre a pergunta e todos os filmes
        similaridades = cosine_similarity(vetor_pergunta, vetores_filmes)[0]
        
        # Adiciona os scores de proximidade ao DataFrame
        df_filmes['proximidade'] = similaridades
        
        # Ordena do mais parecido para o menos parecido e pega os 2 melhores
        recomendacoes = df_filmes.sort_values(by='proximidade', ascending=False).head(2)
        
        st.markdown("### 🍿 Recomendações da IA:")
        
        for idx, linha in recomendacoes.iterrows():
            with st.container():
                st.subheader(f"🎥 {linha['titulo']}")
                st.write(f"**Sinopse:** {linha['descricao']}")
                st.caption(f"Grau de proximidade semântica: {linha['proximidade']:.2f}")
                st.markdown("---")
