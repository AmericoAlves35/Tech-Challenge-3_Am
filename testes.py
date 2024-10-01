import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
from scipy.sparse import csr_matrix

# Carregar o dataset (substitua o caminho pelo seu arquivo CSV)
data = pd.read_csv('Data/IMDB TMDB Movie Metadata Big Dataset (1M).csv')

# Exibir as primeiras linhas do dataset para verificar se foi carregado corretamente
#print(data.head())


#O parâmetro low_memory=False ajudou a otimizar a leitura de dados grandes, enquanto o parâmetro dtype={'adult': str} garantiu que a coluna adult fosse tratada como string, evitando o erro de tipos misturados.
data = pd.read_csv('Data/IMDB TMDB Movie Metadata Big Dataset (1M).csv', low_memory=False, dtype={'adult': str})

# Passo 1: Amostragem
# Obter 50% dos dados
data_sampled = data.sample(frac=0.01, random_state=42)  # random_state para reprodutibilidade

# Passo 2: Remoção de Colunas Desnecessárias
# Defina as colunas que você deseja remover
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

# Agora você pode continuar com sua análise usando data_cleaned
print(data_cleaned.info())  # Verificar as colunas restantes e o tamanho do DataFrame

# Classificar os filmes pela popularidade (número de votos)
top_movies = data.sort_values(by='vote_count', ascending=False)

# Exibir os 10 filmes mais populares
print(top_movies[['title', 'vote_average', 'vote_count']].head(10))

# Gráfico de barras dos filmes mais populares
top_movies.head(10).plot(kind='bar', x='title', y='vote_count', color='blue')
plt.title('Top 10 Filmes Mais Populares')
plt.ylabel('Número de Avaliações')
plt.xlabel('Filme')
plt.tight_layout()  # Ajusta automaticamente a posição dos elementos
plt.show()
#plt.draw()  # Desenha o gráfico sem bloquear a execução
#plt.pause(0.005)  # Pausa por um curto intervalo, permitindo que o gráfico apareça

# Defina features_matrix com colunas que você deseja usar
# Por exemplo, use 'genres_list' e 'Director' juntamente com outras colunas
data['genres_list'] = data['genres_list'].apply(lambda x: x.split(','))  # Convertendo para listas, se necessário
data['Director'] = data['Director'].astype(str)  # Garantir que a coluna seja de tipo string

# Substituir valores NaN por uma lista vazia ou uma string vazia
data['genres_list'] = data['genres_list'].fillna('')  # Substitui NaN por string vazia

# Converter a string em uma lista de gêneros
data['genres_list'] = data['genres_list'].apply(lambda x: x.split(',') if isinstance(x, str) else [])

# Garantir que a coluna 'Director' seja do tipo string
data['Director'] = data['Director'].astype(str)

# Criar a matriz de recursos
features_matrix = data[['popularity', 'AverageRating', 'vote_average', 'genres_list', 'Director']]

# Criar uma cópia do DataFrame
features_matrix = features_matrix.copy()

# Remover linhas com valores nulos usando loc
features_matrix = features_matrix.loc[features_matrix.notnull().all(axis=1)]

# Passo 1: Adicionar a coluna user_rating
data['user_rating'] = 0  # Inicialmente, todas as avaliações são 0
# Substituir valores nulos por 0 na coluna 'user_rating'
data['user_rating'] = data['user_rating'].fillna(0)  # Sem o inplace=True


# simulação o feedback dos usuários, onde podem dar uma nota para os filmes. Com isso, podemos refinar as recomendações.
# Exemplo de feedback simulado do usuário com filmes reais do IMDB presentes na base de dados.
user_feedback = {
    'title': ['The Dark Knight', 'Inception', 'Avatar'],  # Títulos reais dos filmes
    'rating': [0, 0, 5]  # Notas dadas pelo usuário (de 1 a 5)
}

# Converter o feedback em um DataFrame
feedback_df = pd.DataFrame(user_feedback)

# Função para atualizar a matriz de avaliações com o feedback do usuário
def update_ratings_matrix(data, feedback_df):
    for index, row in feedback_df.iterrows():
        # Atualizar a nota do filme na base de dados
        data.loc[data['title'] == row['title'], 'user_rating'] = row['rating']
    return data

# Atualizar a coluna 'user_rating' com valores nulos substituídos por 0
data['user_rating'] = data['user_rating'].fillna(0)


# Função para calcular a média ponderada, ignorando notas 0
def calcular_media_ponderada(row):
    # Pesos para a média ponderada
    peso_vote_average = 0.5  # Ajuste conforme necessário
    peso_user_rating = 0.5

    # Se a user_rating for 0, não consideramos
    if row['user_rating'] == 0:
        return row['vote_average']

    return (peso_vote_average * row['vote_average'] + peso_user_rating * row['user_rating']) / (peso_vote_average + peso_user_rating)

# Criar uma nova coluna para a média ponderada
data['media_ponderada'] = data.apply(calcular_media_ponderada, axis=1)

# Atualizar as notas dos filmes com base no feedback
data_atualizada = update_ratings_matrix(data, feedback_df)

# Mostrar as atualizações
print(data_atualizada[['title', 'user_rating']].head())

# Uso da matriz de características (features) de filmes
features_matrix = data[['vote_average', 'user_rating', 'popularity']].copy()  # cópia explícita do DataFrame

# Remover linhas com valores nulos
#features_matrix = features_matrix.dropna()  # Remover NaNs sem o inplace=True para evitar o aviso

# Reduzir a amostra para 1000 filmes
features_matrix = features_matrix.sample(n=1000, random_state=42)

# Remover linhas com valores nulos
features_matrix = features_matrix.dropna()  # Remover NaNs

# Criar a matriz esparsa
sparse_features_matrix = csr_matrix(features_matrix)

# Calcular a similaridade dos filmes
similarity_matrix = cosine_similarity(sparse_features_matrix)

# Criar um DataFrame para facilitar a consulta
similarity_df = pd.DataFrame(similarity_matrix, index=data['title'][features_matrix.index], columns=data['title'][features_matrix.index])

from sklearn.preprocessing import MultiLabelBinarizer

# Supondo que você já tenha carregado o DataFrame `data`
# Limpar e garantir que a coluna 'Director' seja uma lista de diretores
data['Director'] = data['Director'].fillna('').apply(lambda x: [d.strip() for d in x.split(',')] if x else [])

# Usar MultiLabelBinarizer para transformar a coluna 'Director'
mlb = MultiLabelBinarizer()
director_dummies = pd.DataFrame(mlb.fit_transform(data['Director']), columns=mlb.classes_, index=data.index)

# Juntar as colunas binárias de volta ao DataFrame original
data = pd.concat([data, director_dummies], axis=1)

# Verifique as primeiras linhas do DataFrame atualizado
print(data.head())
