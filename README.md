# 🎬 Assistente de Recomendação com Busca Semântica (RAG)

Este projeto implementa um sistema inteligente de recomendação utilizando conceitos de inteligência artificial generativa e recuperação de dados baseada em contexto (Retrieval-Augmented Generation - RAG).

## 🚀 Como o Sistema Funciona

1. **Vetorização Semântica (Embeddings):** Transforma textos complexos (sinopses de filmes) em coordenadas vetoriais numéricas utilizando o modelo de linguagem `SentenceTransformer (all-MiniLM-L6-v2)`.
2. **Cálculo de Proximidade:** Quando o usuário faz uma pergunta em linguagem natural, o sistema vetoriza a dúvida e aplica o cálculo de **Similaridade de Cosseno** (via Scikit-Learn) direto na memória.
3. **Entrega Inteligente:** Retorna em tempo real as produções que mais compartilham contexto semântico com a busca do usuário, ignorando a necessidade de palavras-chave exatas.

---

## 🛠️ Tecnologias Utilizadas

* **Linguagem:** Python
* **Inteligência Artificial & NLP:** Sentence-Transformers (Hugging Face)
* **Modelagem Matemática:** Scikit-Learn (Cosine Similarity)
* **Manipulação de Dados:** Pandas
* **Interface Gráfica:** Streamlit
