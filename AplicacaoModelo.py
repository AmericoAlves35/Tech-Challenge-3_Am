import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

# Carregar o dataset (substitua o caminho pelo seu arquivo CSV)
data = pd.read_csv('Data/IMDB TMDB Movie Metadata Big Dataset (1M).csv', low_memory=False)

data_sampled = data.sample(frac=1.0, random_state=42)  # random_state para reprodutibilidade

# Exibir as primeiras linhas do dataset para verificar se foi carregado corretamente
print(data.head())
print(data.info())  # Para ver informações sobre as colunas e tipos de dados


# Passo 2: Remoção de Colunas Desnecessárias
# Defina as colunas que você deseja remover
colunas_a_remover = [
    'id',
    'homepage',
    'backdrop_path',
    'poster_path',
    'tagline',
    'original_language',
    'production_companies',
    'production_countries',
    'certificate',
    'AverageRating',
    'status',
    'revenue',
    'budget'
]

# Remover as colunas
data_cleaned = data_sampled.drop(columns=colunas_a_remover, errors='ignore')


# Lidar com valores nulos
# Remover linhas com valores nulos
data_cleaned = data_cleaned.dropna()

# Ou, se preferir, você pode preencher os valores nulos com a média (por exemplo)
# data_cleaned['popularidade'] = data_cleaned['popularidade'].fillna(data_cleaned['popularidade'].mean())

# Verificar o resultado da limpeza
print(data_cleaned.info())

# Passo 3: Amostragem
# Amostrar 10% dos dados para testes
data_sampled = data_cleaned.sample(frac=0.1, random_state=42)  # 10% da amostra, pode ajustar conforme necessidade

# Verificar o tamanho da amostra
print(data_sampled.shape)

# Passo 4: Criar a Matriz de Características
# Selecionar as colunas de interesse (popularidade e avaliação)
feature_columns = ['popularity', 'vote_average']

# Criar a matriz de características
feature_matrix = data_sampled[feature_columns]

# Verificar a matriz de características
print(feature_matrix.head())


# Passo 5: Calcular a Similaridade
# Calcular a similaridade de cosseno entre os filmes
similarity_matrix = cosine_similarity(feature_matrix)

# Verificar a matriz de similaridade
print(similarity_matrix)
import numpy as np


# Passo 6: Sistema de Recomendação

# Ordenar os filmes por avaliação média e selecionar os 10 melhores
top_10_avaliados = data_cleaned.sort_values('vote_average', ascending=False).head(10)

# Criar o gráfico de barras
plt.figure(figsize=(10, 6))
plt.barh(top_10_avaliados['title'], top_10_avaliados['vote_average'], color='skyblue')
plt.xlabel('Avaliação Média')
plt.ylabel('Filme')
plt.title('Top 10 Filmes com as Melhores Avaliações')
plt.gca().invert_yaxis()  # Inverter o eixo Y para exibir o filme com a maior avaliação no topo
plt.tight_layout()  # Ajusta automaticamente a posição dos elementos
plt.show()

def recomendar_filmes(titulo_filme, data, similarity_matrix, top_n=10):
    idx = data[data['title'] == titulo_filme].index[0]

    print(f"Índice: {idx}, Tamanho da matriz: {similarity_matrix.shape[0]}")  # Debug

    if idx >= similarity_matrix.shape[0]:
        print(f"Índice {idx} fora dos limites. Verifique o título do filme.")
        return []

    similarity_scores = list(enumerate(similarity_matrix[idx]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    while True:
        # Verificar se o filme existe no dataset
        if titulo_filme not in data['title'].values:
            print("Filme não encontrado, insira apenas títulos existentes no Dataset.")
            # Perguntar se o usuário deseja tentar novamente
            continuar = input("Deseja inserir outro título? (s/n): ").strip().lower()
            if continuar == 's':
                titulo_filme = input("Por favor, insira o título do filme novamente: ")
                continue
            else:
                return "Saindo do sistema de recomendações."

        # Obter o índice do filme no DataFrame
        idx = data[data['title'] == titulo_filme].index[0]

        # Obter as similaridades do filme com todos os outros
        similarity_scores = list(enumerate(similarity_matrix[idx]))

        # Classificar os filmes com base na similaridade
        sorted_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

        # Obter os índices dos top_n filmes mais semelhantes (ignorando o próprio filme)
        top_filmes_indices = [i[0] for i in sorted_scores[1:top_n + 1]]

        # Retornar os títulos dos filmes recomendados
        filmes_recomendados = data['title'].iloc[top_filmes_indices]
        return filmes_recomendados


# Exemplo de chamada do loop de recomendação
#tentar_novamente(data_cleaned, similarity_matrix)


# Exemplo de uso

filme_escolhido = input("Digite o título de um filme para obter recomendações: ")
recomendacoes = recomendar_filmes(filme_escolhido, data_cleaned, similarity_matrix, top_n=10)

if isinstance(recomendacoes, str):
    print(recomendacoes)  # Exibe a mensagem de erro ou saída
else:
    print(f"\nFilmes recomendados para '{filme_escolhido}':")
    for filme in recomendacoes:
        print(f"- {filme}")

print(recomendacoes)
# Criar o gráfico de barras para os filmes recomendados
plt.figure(figsize=(10, 6))
plt.barh(recomendacoes, range(1, len(recomendacoes) + 1), color='lightgreen')
plt.xlabel('Ranking de Recomendação')
plt.ylabel('Filme')
plt.title(f'Filmes Recomendados para: {filme_escolhido}')
plt.gca().invert_yaxis()  # Inverter o eixo Y para exibir o mais recomendado no topo
plt.tight_layout()  # Ajusta automaticamente a posição dos elementos
plt.show()

def adicionar_avaliacao(titulo, nota, dados):
    # Verifica se o filme está no dataset
    if titulo not in dados['title'].values:
        return "Filme não encontrado, insira apenas titulos existentes no Dataset"

    # Adiciona a nova avaliação ao DataFrame
    dados.loc[dados['title'] == titulo, 'user_rating'] = nota
    print(f"Avaliação de {nota} adicionada ao filme '{titulo}'.")

# Atualizar a matriz de características
def atualizar_matriz(dados):
    # Cria a nova matriz de características usando a nova coluna de avaliações
    return dados[['popularity', 'user_rating']]

# Recalcular a matriz de similaridade
def recalcular_similaridade():
    # Certifique-se de que não há NaNs na matriz de características
    if data.isnull().values.any():
        raise ValueError("A matriz de dados contém NaN. Por favor, trate os valores ausentes antes de recalcular.")

    # Calcula a similaridade do cosseno
    matriz_similaridade = cosine_similarity(data)
    print("Similaridade recalculada com sucesso!")
    return matriz_similaridade


# Chame a função para recalcular a similaridade
#try:
    #recalcular_similaridade()
#except ValueError as e:
    #print(e)

# Exemplo de uso
#titulo_novo = input("Digite o título do filme que deseja avaliar: ")
#nota_nova = float(input("Digite sua avaliação para o filme (0 a 10): "))
#adicionar_avaliacao(titulo_novo, nota_nova, data_cleaned)

# Atualizar a matriz de características e recalcular similaridade
#matriz_atualizada = atualizar_matriz(data_cleaned)
#similarity_matrix = recalcular_similaridade(matriz_atualizada)

#print("Matriz de similaridade recalculada com base nas novas avaliações.")


# Opção de Treinamento
def opcao_treinamento(data):
    resposta = input("Deseja treinar o sistema com novas avaliações? (sim/não): ").strip().lower()

    if resposta == 'sim':
        titulo_novo = input("Digite o título do filme que deseja avaliar: ")
        nota_nova = float(input("Digite sua avaliação para o filme (0 a 10): "))
        adicionar_avaliacao(titulo_novo, nota_nova, data)

        # Atualizar a matriz de características e recalcular similaridade
        matriz_atualizada = atualizar_matriz(data)
        similarity_matrix = recalcular_similaridade(matriz_atualizada)
        print("Matriz de similaridade recalculada com base nas novas avaliações.")
    else:
        print("Prosseguindo sem treinamento.")


# Chame a função para decidir se o usuário deseja treinar
#opcao_treinamento(data_cleaned)




# Dividir o dataset em treino e teste
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Função para calcular as métricas de avaliação
# Função ajustada para calcular as métricas de avaliação
def avaliar_recomendacoes(titulo_filme, data_teste, similarity_matrix, top_n=10):
    # Obter as recomendações para o filme do conjunto de teste

    if not data_teste[data_teste['title'] == titulo_filme].empty:
        filme_real_idx = data_teste[data_teste['title'] == titulo_filme].index[0]
    else:
        print(f"Filme '{titulo_filme}' não encontrado no dataset de teste.")
        return

    filmes_recomendados = recomendar_filmes(titulo_filme, data_teste, similarity_matrix, top_n)

    # Verificar o próprio filme no dataset de teste
    filme_real_idx = data_teste[data_teste['title'] == titulo_filme].index[0]

    # Obter os filmes mais semelhantes ao filme atual (esperado)
    filmes_esperados = similarity_matrix[filme_real_idx].argsort()[-(top_n + 1):][::-1][1:top_n + 1]

    # Converter os filmes recomendados e esperados em sets
    filmes_recomendados_set = set(filmes_recomendados)
    filmes_esperados_set = set(data_teste['title'].iloc[filmes_esperados])

    # Calcular true positives, precision, recall e F1-score
    true_positive = len(filmes_recomendados_set.intersection(filmes_esperados_set))
    precision = true_positive / len(filmes_recomendados_set) if len(filmes_recomendados_set) > 0 else 0
    recall = true_positive / len(filmes_esperados_set) if len(filmes_esperados_set) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1


# Exemplo de uso para um filme
titulo_filme ='Inception'
precision, recall, f1 = avaliar_recomendacoes(titulo_filme, test_data, similarity_matrix)
print(f"Precisão: {precision:.2f}, Recall: {recall:.2f}, F1-Score: {f1:.2f}")



def validar_modelo(data, similarity_matrix, top_n=10, k=5):
    kf = KFold(n_splits=k)
    precisions, recalls, f1_scores = [], [], []

    for train_index, test_index in kf.split(data):
        train_data = data.iloc[train_index]
        test_data = data.iloc[test_index]

        for titulo_filme in test_data['title'].values:
            precision, recall, f1 = avaliar_recomendacoes(titulo_filme, test_data, similarity_matrix, top_n)
            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1)

    return {
        'media_precision': sum(precisions) / len(precisions),
        'media_recall': sum(recalls) / len(recalls),
        'media_f1': sum(f1_scores) / len(f1_scores)
    }


# Exemplo de uso
resultados_validacao = validar_modelo(data, similarity_matrix, top_n=10, k=5)
print(f"Média Precisão: {resultados_validacao['media_precision']:.2f}, "
      f"Média Recall: {resultados_validacao['media_recall']:.2f}, "
      f"Média F1-Score: {resultados_validacao['media_f1']:.2f}")
