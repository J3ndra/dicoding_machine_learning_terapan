# -*- coding: utf-8 -*-
"""submisison_recommendation_system.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1aew8S3G8aMSo3aGyieXZUK46xXKuJPnC

# Recommendation System

Oleh: Junianto Endra Kartika

## Import Library
"""

import re

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras import layers, regularizers

"""## Data Loading

Dataset yang digunakan bernama [MovieLens](https://grouplens.org/datasets/movielens/) dengan total data yang ada adalah 100 ribu data.

### Users
"""

# Memuat data user

user_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
users = pd.read_csv('../Dataset/ml-100k/u.user', sep='|', header=None, names=user_cols, engine='python', encoding='latin-1')
users.head()

users.info()

"""Hasil menunjukkan data `users` memiliki 5 kolom dan total 943 data dengan tidak ada yang kosong (`null`)."""

# Menampilkan jumlah pengguna

n_users = users.shape[0]
print(f'Jumlah pengguna: {n_users}')

"""### Genres"""

# Memuat data genre

genre_cols = ['genre_name', 'genre_id']
genres = pd.read_csv('../Dataset/ml-100k/u.genre', sep='|', header=None, names=genre_cols, engine='python', encoding='latin-1')
genres.head()

genres.info()

"""Hasil menunjukkan data `genres` memiliki 2 kolom dan total 19 data dengan tidak ada yang kosong (`null`)."""

# Menampilkan jumlah genre

n_genres = genres.shape[0]
print(f'Jumlah genre: {n_genres}')

"""### Movies"""

# Memuat data movie

movie_cols = ['movie_id', 'movie_title', 'release_date', 'video_release_date', 'IMDb_URL']
movie_cols += genres['genre_name'].tolist()

movies = pd.read_csv('../Dataset/ml-100k/u.item', sep='|', header=None, names=movie_cols, engine='python', encoding='latin-1')

# Menampilkan 5 data movie pertama
movies.head()

# Menampilkan 5 data movie terakhir
movies.tail()

movies.info()

"""Hasil menunjukkan data `movies` memiliki 24 kolom dan total 1682 data dengan tidak ada yang kosong (`null`)."""

# Menampilkan jumlah movie

n_movies = movies.shape[0]
print(f'Jumlah movies: {n_movies}')

"""### Ratings"""

# Memuat data rating

rating_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']

ratings_base = pd.read_csv('../Dataset/ml-100k/ua.base', sep='\t', header=None, names=rating_cols, engine='python', encoding='latin-1')
ratings_test = pd.read_csv('../Dataset/ml-100k/ua.test', sep='\t', header=None, names=rating_cols, engine='python', encoding='latin-1')

ratings_base.head()

ratings_base.info()
ratings_test.info()

"""Hasil menunjukkan data `ratings_base` dan `ratings_test` memiliki 4 kolom dan total 90570 data pada `ratings_base` dan 9430 data pada `ratings_test` dengan tidak ada yang kosong (`null`)."""

# Menampilkan jumlah rating

n_train = ratings_base.values
n_test = ratings_test.values

print(f'Jumlah rating train: {n_train.shape[0]}')
print(f'Jumlah rating test: {n_test.shape[0]}')

"""Pada proyek ini, akan terdapat beberapa data yang akan digunakan. Yaitu

1. Users:
    - `user_id`
    - `age`
    - `occupation`
    - `zip_code`
2. Genres
    - `genre_id`
    - `genre_name`
3. Movies
    - `movie_id`
    - `movie_title`
    - `release_date`
    - `video_release_date`
    - `IMDb_URL`
    - One-hot genres encoding
4. Ratings
    - `user_id`
    - `movie_id`
    - `rating`
    - `unix_timestamp`

## EDA

### Users
"""

# Menggabungkan peringkat dari kumpulan data dasar dan uji
all_ratings = pd.concat([ratings_base, ratings_test])

# Menghitung rating per pengguna
ratings_per_user = all_ratings.groupby('user_id')['rating'].count()

# Menghitung statistik
min_ratings = ratings_per_user.min()
max_ratings = ratings_per_user.max()
mean_ratings = ratings_per_user.mean()
median_ratings = ratings_per_user.median()
std_ratings = ratings_per_user.std()

# Menampilkan statistik
plt.figure(figsize=(12, 8))

n, bins, patches = plt.hist(ratings_per_user, bins=30, alpha=0.7, color='skyblue', edgecolor='black')

plt.axvline(mean_ratings, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_ratings:.1f}')
plt.axvline(median_ratings, color='green', linestyle='dashed', linewidth=2, label=f'Median: {median_ratings:.1f}')

stats_text = f'Min: {min_ratings}\nMax: {max_ratings}\nStd: {std_ratings:.1f}'
plt.annotate(stats_text, xy=(0.95, 0.95), xycoords='axes fraction',
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8),
             ha='right', va='top')

plt.title('Distribution of Ratings per User', fontsize=15)
plt.xlabel('Number of Ratings', fontsize=12)
plt.ylabel('Number of Users', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Menampilkan informasi pengguna
print(f"Total number of users: {len(ratings_per_user)}")
print(f"Total number of ratings: {len(all_ratings)}")
print(f"Average number of ratings per user: {mean_ratings:.2f}")
print(f"Maximum number of ratings by a user: {max_ratings}")
print(f"Minimum number of ratings by a user: {min_ratings}")

"""Visualisasi di atas menunjukkan dari jumlah 943 *user*, rata-rata *user* memiliki jumlah rating `106` dengan *user* yang memiliki jumlah rating tertinggi berjumlah `737` dan jumlah rating terendah adalah `20`.

### Genres
"""

# Menghitung jumlah film untuk setiap genre
genre_counts = {}
genre_columns = [col for col in movies.columns if col not in ['movie_id', 'movie_title', 'release_date', 'video_release_date', 'IMDb_URL']]
for genre in genre_columns:
    genre_counts[genre] = movies[genre].sum()

# Menghitung total film untuk setiap genre
genre_distribution = pd.Series(genre_counts).sort_values(ascending=False)

# Menampilkan distribusi genre
plt.figure(figsize=(12, 8))
ax = sns.barplot(x=genre_distribution.values, y=genre_distribution.index, hue=genre_distribution.index, palette='viridis', legend=False)

for i, v in enumerate(genre_distribution.values):
    ax.text(v + 5, i, f"{int(v)}", va='center')

plt.title('Distribution of Movie Genres', fontsize=15)
plt.xlabel('Number of Movies', fontsize=12)
plt.ylabel('Genre', fontsize=12)

plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.show()

# Menampilkan informasi genre
print(f"Total number of genre assignments: {sum(genre_distribution.values)}")
print(f"Average genres per movie: {sum(genre_distribution.values) / n_movies:.2f}")
print(f"Most common genre: {genre_distribution.index[0]} ({int(genre_distribution.values[0])} movies)")
print(f"Least common genre: {genre_distribution.index[-1]} ({int(genre_distribution.values[-1])} movies)")

"""Visualisasi di atas menunjukkan bahwa *genre* `drama` merupakan *genre* dengan jumlah movie terbanyak. Terdapat *genre* `unknown` yang dimana *genre* tersebut adalah *genre* yang kotor, maka dari itu proyek ini akan menghapus *genre* tersebut."""

# Mencari film dengan genre 'unknown'
unknown_movies = movies[movies['unknown'] == 1]
print(f"Number of movies with 'unknown' genre: {len(unknown_movies)}")

# Mencari film dengan genre 'unknown' dan genre lain
genre_columns = [col for col in movies.columns if col not in ['movie_id', 'movie_title', 'release_date', 'video_release_date', 'IMDb_URL', 'unknown']]
has_other_genres = unknown_movies[genre_columns].sum(axis=1) > 0

# Menampilkan film dengan genre 'unknown' dan genre lain
unknown_movies_details = unknown_movies[['movie_id', 'movie_title', 'release_date']]
print("\nDetails of unknown genre movies:")
print(unknown_movies_details)

# Menampilkan jumlah film dengan genre 'unknown' dan genre lain
print(f"\nMovies with 'unknown' genre that also have other genres: {has_other_genres.sum()}")
print(f"Movies with ONLY 'unknown' genre: {(~has_other_genres).sum()}")

"""Karena hanya 2 movie yang memiliki genre `unknown` dan movie tersebut tidak memiliki genre yang lain, maka movie tersebut akan dihapus dari data untuk menjaga keseimbangan data."""

# Menghapus film dengan genre 'unknown' yang tidak memiliki genre lain
movie_ids_to_remove = unknown_movies[~has_other_genres]['movie_id'].tolist()
movies = movies[~movies['movie_id'].isin(movie_ids_to_remove)]

# Menghapus kolom 'unknown' dari DataFrame movies
movies = movies.drop('unknown', axis=1)

# Menghapus genre 'unknown' dari DataFrame genres
genre_columns = [col for col in movies.columns if col not in ['movie_id', 'movie_title', 'release_date', 'video_release_date', 'IMDb_URL']]

# Menghapus kolom 'unknown' dari genre_cols
if 'unknown' in movie_cols:
    movie_cols.remove('unknown')

# Menghapus genre 'unknown' dari DataFrame genres
genres = genres[genres['genre_name'] != 'unknown']

print(f"Menghapus {len(movie_ids_to_remove)} movies dengen gener 'unknown'.")
print(f"Jumlah movies setelah penghapusan: {movies.shape[0]}")
print(f"Jumlah genres setelah penghapusan 'unknown': {len(genres)}")

"""Setelah menghapus *genre* `unknown` dan *movie* yang memiliki genre `unknown`. Jumlah *movie* sebelumnya berjumlah `1682` menjadi `1680` dan jumlah *genre* yang sebelumnya `19` menjadi `18`.

### Ratings
"""

plt.figure(figsize=(12, 8))

# Menampilkan distribusi rating
rating_counts = all_ratings['rating'].value_counts().sort_index()
ax = sns.barplot(x=rating_counts.index, y=rating_counts.values, palette='viridis', legend=False)

for i, count in enumerate(rating_counts.values):
	ax.text(i, count + 500, f"{count:,}", ha='center', fontsize=10)

# Menambahkan statisik
mean_rating = all_ratings['rating'].mean()
median_rating = all_ratings['rating'].median()
total_ratings = len(all_ratings)

plt.axvline(x=mean_rating-1, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_rating:.2f}')
plt.axvline(x=median_rating-1, color='green', linestyle='--', linewidth=2, label=f'Median: {median_rating:.1f}')

plt.title('Distribution of Movie Ratings', fontsize=16)
plt.xlabel('Rating', fontsize=12)
plt.ylabel('Number of Ratings', fontsize=12)
plt.xticks(ticks=range(5), labels=['1', '2', '3', '4', '5'])

plt.figtext(0.91, 0.85, f"Total ratings: {total_ratings:,}\nMean: {mean_rating:.2f}\nMedian: {median_rating:.1f}",
			bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8),
			horizontalalignment='right')

plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.show()

"""Visualisasi di atas menunjukkan distribusi *rating* dimana *rating* terendah ada di `1` dan *rating* tertinggi ada di `5` dengan rata-rata *rating* terdapat di `4`.

## Data Preprocessing

### Menggabungkan Data

#### Content based data

Karena proyek ini akan menggunakan [**TF-IDF**](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html), kita akan menggabungkan semua *One-hot Encoded genre* yang terdapat pada data *movie* menjadi satu kolom bernama *genres* seperti `Animation Children's Comedy` atau `Action Advanture Thriller`.
"""

# Membuat fungsi untuk menyiapkan data untuk rekomendasi berbasis konten
def prepare_content_based_data():
    movie_features = movies.copy()

    # Membuat variable genre untuk setiap genre yang ada
    genre_columns = ['Action', 'Adventure', 'Animation', "Children's", 'Comedy', 'Crime',
                     'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical',
                     'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']

    # Mengubah genre menjadi one-hot encoding
    content_df = movie_features[['movie_id', 'movie_title'] + genre_columns].copy()

    # Ubah kolom one-hot encoded genre menjadi kolom teks tunggal untuk pemrosesan TF-IDF
    content_df['genres'] = content_df[genre_columns].apply(
        lambda row: ' '.join([genre for genre, val in zip(genre_columns, row) if val == 1]),
        axis=1
    )

    # Mereset indeks untuk menghindari masalah saat menggabungkan dengan rating
    content_df = content_df.reset_index(drop=True)

    return content_df, genre_columns

# Memanggil fungsi dan menampilkan data
content_data, genre_columns = prepare_content_based_data()
content_data.head()

"""#### Collaborative data

Fungsi ini akan menyiapkan data untuk pendekatan *Collaborative Filtering*, dengan cara mencari kesamaan antara pengguna berdasarkan rating film yang telah mereka berikan.
"""

# membuat fungsi untuk menyiapkan data untuk rekomendasi berbasis kolaboratif
def prepare_collaborative_data(ratings_data):
    cf_data = ratings_data.copy()

    # Menggabungkan dengan data pengguna
    cf_data_with_users = cf_data.merge(users[['user_id', 'age', 'sex', 'occupation']], on='user_id', how='left')

    # Menggabungkan dengan data film
    cf_data_with_movies = cf_data_with_users.merge(
        movies[['movie_id', 'movie_title']],
        on='movie_id',
        how='left'
    )

    # Hapus baris dengan movie_title yang null
    cf_data_with_movies = cf_data_with_movies.dropna(subset=['movie_title'])

    # Update cf_data untuk hanya berisi movie_ids yang memiliki movie_title
    cf_data = cf_data[cf_data['movie_id'].isin(cf_data_with_movies['movie_id'])]

    # Membuat matriks interaksi pengguna-item (matriks rating pengguna-item)
    user_item_matrix = cf_data.pivot(index='user_id', columns='movie_id', values='rating')

    # Mendapatkan daftar semua ID pengguna dan ID film
    user_ids = cf_data['user_id'].unique().tolist()
    movie_ids = cf_data['movie_id'].unique().tolist()

    # Membuat pemetaan dari ID asli ke ID urutan (0 sampai n-1)
    user_to_user_encoded = {x: i for i, x in enumerate(user_ids)}
    movie_to_movie_encoded = {x: i for i, x in enumerate(movie_ids)}

    # Membuat pemetaan dari ID urutan kembali ke ID asli
    user_encoded_to_user = {i: x for i, x in enumerate(user_ids)}
    movie_encoded_to_movie = {i: x for i, x in enumerate(movie_ids)}

    # Mengkonversi rating ke skala 0-1 (normalisasi)
    cf_data['rating'] = cf_data['rating'] / 5.0

    # Menggunakan ID terenkode untuk pengguna dan film
    cf_data['user_encoded'] = cf_data['user_id'].map(user_to_user_encoded)
    cf_data['movie_encoded'] = cf_data['movie_id'].map(movie_to_movie_encoded)

    return cf_data_with_movies, user_item_matrix, cf_data, user_encoded_to_user, movie_encoded_to_movie, len(user_ids), len(movie_ids)

# Memanggil fungsi
ratings_combined = pd.concat([ratings_base, ratings_test], ignore_index=True)
cf_merged_data, user_item_matrix, cf_data_processed, user_encoded_to_user, movie_encoded_to_movie, num_users, num_movies = prepare_collaborative_data(ratings_combined)

# Menampilkan 5 data teratas `cf_data_processed`
cf_data_processed.head()

"""## Data Preparation

### Mengatasi missing value
"""

# Mengecek apakah ada nilai yang hilang dalam data
content_data.isnull().sum()

# Mengecek apakah ada nilai yang hilang dalam data
cf_data_processed.isnull().sum()

"""### Conten-Based data preparation

Data kolom `genres` yang sebelumnya diolah akan dirubah menggunakan **TF-IDF**.
"""

# Membuat matriks TF-IDF untuk genre
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(content_data['genres'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Menampilkan matriks TF-IDF
pd.DataFrame(
    tfidf_matrix.todense(),
    columns=tfidf.get_feature_names_out(),
    index=content_data['movie_title']
).sample(10, axis=1).sample(5, axis=0)

"""### Collaborative data preparation

Data akan dibagi menjadi data latih dan data uji menggunakan [*train_test_split*](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) dari data yang telah dibagun pada bagan *data preprocessing*.
"""

# Memisahakan data menjadi data pelatihan dan validasi
x = cf_data_processed[['user_encoded', 'movie_encoded']].values
y = cf_data_processed['rating'].values

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)
print(f"Training samples: {len(x_train)}, Validation samples: {len(x_val)}")

"""## Content Based Filtering"""

# Membuat fungsi untuk merekomendasikan film berdasarkan judul
def recommend(title, cosine_sim=cosine_sim, k=10):
    # Membuat pemetaan dari judul film ke indeks
    indices = pd.Series(content_data.index, index=content_data['movie_title']).drop_duplicates()

    try:
        # Mendapatkan indeks film yang sesuai dengan judul
        idx = indices[title]

        # Memeriksa apakah indeks berada dalam rentang yang valid
        if idx >= len(cosine_sim):
            print(f"Movie index {idx} is out of bounds (max valid index: {len(cosine_sim)-1})")
            return []

        print(f"Movie index: {idx}")

        # Mendapatkan skor kesamaan untuk semua film dengan film target
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:k+1]  # Mendapatkan k film serupa teratas kecuali film itu sendiri

        # Mendapatkan genre film target
        target_genres = set([genre for genre in genre_columns if content_data.iloc[idx][genre] == 1])

        # Menghitung precision@k
        relevant_count = 0

        # Mendapatkan indeks film dan menghitung precision@k
        recommendations = []
        for i, score in sim_scores:
            movie_genres = set([genre for genre in genre_columns if content_data.iloc[i][genre] == 1])
            # Dianggap relevan jika ada kesamaan genre
            is_relevant = len(target_genres.intersection(movie_genres)) > 0
            if is_relevant:
                relevant_count += 1

            # Mendapatkan genre utama (pertama) dari film yang direkomendasikan
            main_genre = next(iter(movie_genres)) if movie_genres else "Unknown"

            recommendations.append({
                'movie_id': content_data.iloc[i]['movie_id'],
                'title': content_data.iloc[i]['movie_title'],
                'genre': main_genre,
                'similarity': round(score, 4),
                'is_relevant': is_relevant
            })

        precision_k = relevant_count / k if k > 0 else 0

        return {
            'recommendations': recommendations,
            'precision_at_k': precision_k,
            'k': k,
            'target_genres': target_genres
        }
    except KeyError:
        print(f"Movie '{title}' not found in the dataset.")
        return []
    except IndexError as e:
        print(f"Index error: {e}. Movie index {idx} may be out of range.")
        return []
    except Exception as e:
        print(f"Error recommending movies: {str(e)}")
        return []

# Membuat fungsi untuk menampilkan rekomendasi film dalam format tabel
def display_recommendations(recommendations_result, title):
    if not isinstance(recommendations_result, dict) or 'recommendations' not in recommendations_result:
        print("No valid recommendations to display.")
        return

    # Membuat DataFrame untuk menampilkan rekomendasi
    df = pd.DataFrame(recommendations_result['recommendations'])

    # Menambahkan nomor indeks untuk tampilan yang lebih baik
    df.index = range(1, len(df) + 1)

    # Mengatur urutan kolom
    df = df[['title', 'genre', 'similarity']]

    print(f"\nRecommendations for '{title}':")
    print(f"Target genres: {', '.join(recommendations_result['target_genres'])}")
    print(f"\nPrecision@{recommendations_result['k']}: {recommendations_result['precision_at_k']:.2f}")
    print("\n" + "-"*70)
    print(df.to_string())
    print("-"*70)

    return df

# Membuat fungsi untuk menampilkan rekomendasi film
def visualize_recommendation(movie_title, recommendations_result, content_data):
    try:
        if not isinstance(recommendations_result, dict) or 'recommendations' not in recommendations_result:
            print("No valid recommendations to visualize.")
            return

        movie_idx = content_data[content_data['movie_title'] == movie_title].index[0]
        source_movie = content_data.iloc[movie_idx]

        # Mendapatkan rekomendasi
        recommendations = recommendations_result['recommendations']

        rec_count = min(5, len(recommendations))
        movie_ids = [source_movie['movie_id']] + [rec['movie_id'] for rec in recommendations[:rec_count]]
        movies_to_compare = content_data[content_data['movie_id'].isin(movie_ids)]

        plt.figure(figsize=(12, 8))
        genre_profiles = movies_to_compare.set_index('movie_title')[genre_columns]

        sns.heatmap(genre_profiles.astype(float), cmap='viridis', cbar_kws={'label': 'Genre Present'})

        plt.title(f'Genre Profile Comparison: {movie_title} and Similar Movies', fontsize=15)
        plt.xlabel('Genres', fontsize=12)
        plt.ylabel('Movies', fontsize=12)
        plt.tight_layout()
        plt.show()

    except IndexError:
        print(f"Movie '{movie_title}' not found.")
        return
    except Exception as e:
        print(f"Error visualizing recommendations: {str(e)}")

"""### Testing"""

# Example of how to use the updated recommend function
movie_title = "Toy Story (1995)"
toy_story_results = recommend(movie_title, k=10)

# Menampilkan rekomendasi dalam format tabel
display_recommendations(toy_story_results, movie_title)

# Memvisualisasikan rekomendasi untuk Toy Story
visualize_recommendation(movie_title, toy_story_results, content_data)

"""Visualisasi menunjukkan bahwa film **Toy Story (1995)** memiliki 3 *genre* yaitu `Comedy, Animation, Children's` dimana film **Aladdin and the King of Thieves (1996)** memiliki 3 *genre* yang sama. Sedangkan film yang lain hanya memiliki 2 *genre* yang sama, maka dari itu *cosine similarity* hanya memiliki score `0.937`."""

# Example of how to use the updated recommend function
movie_title = "You So Crazy (1994)"
you_so_crazy_results = recommend(movie_title, k=10)

# Menampilkan rekomendasi dalam format tabel
display_recommendations(you_so_crazy_results, movie_title)

# Memvisualisasikan rekomendasi untuk Toy Story
visualize_recommendation(movie_title, you_so_crazy_results, content_data)

"""Visualisasi menunjukkan bahwa film **You So Crazy (1994)** memiliki 1 *genre* saja, yaitu `Comedy` dimana semua 10 film memiliki *genre* yang sama. Maka dari itu semua film memiliki score *cosine similarity* `1.0`.

## Collaborative Filtering

Kelas yang akan dibagun akan menggunakan [*Keras Model Class*](https://keras.io/api/models/model/) yang terinspirasi dari [*Collaborative Filtering MovieLens* dari Keras](https://keras.io/examples/structured_data/collaborative_filtering_movielens/).
"""

# Membuat model rekomendasi berbasis kolaboratif menggunakan TensorFlow
class RecommenderNet(tf.keras.Model):
    def __init__(self, num_users, num_movies, embedding_size, **kwargs):
        super(RecommenderNet, self).__init__(**kwargs)
        self.num_users = num_users
        self.num_movies = num_movies
        self.embedding_size = embedding_size

        # User embedding layer
        self.user_embedding = layers.Embedding(
            num_users,
            embedding_size,
            embeddings_initializer='he_normal',
            embeddings_regularizer=regularizers.l2(1e-6),
            name="user_embedding"
        )

        # User bias layer
        self.user_bias = layers.Embedding(
            num_users,
            1,
            embeddings_initializer='zeros',
            name="user_bias"
        )

        # Movie embedding layer
        self.movie_embedding = layers.Embedding(
            num_movies,
            embedding_size,
            embeddings_initializer='he_normal',
            embeddings_regularizer=regularizers.l2(1e-6),
            name="movie_embedding"
        )

        # Movie bias layer
        self.movie_bias = layers.Embedding(
            num_movies,
            1,
            embeddings_initializer='zeros',
            name="movie_bias"
        )

    def call(self, inputs):
        # Ekstrak indeks pengguna dan film dari input
        user_vector = self.user_embedding(inputs[:, 0])
        user_bias = self.user_bias(inputs[:, 0])

        movie_vector = self.movie_embedding(inputs[:, 1])
        movie_bias = self.movie_bias(inputs[:, 1])

        # Menghitung dot product antara embedding pengguna dan film
        dot_user_movie = tf.tensordot(user_vector, movie_vector, 2)

        # Menambahkan semua komponen (dot product + bias)
        x = dot_user_movie + user_bias + movie_bias

        # Menerapkan sigmoid untuk membatasi rating dalam rentang [0, 1]
        return tf.nn.sigmoid(x)

"""*Compiling* terhadap kelas model yang dibangun"""

# Membuat dan mengkompilasi model
model = RecommenderNet(num_users, num_movies, 50)
model.compile(
    loss=tf.keras.losses.MeanSquaredError(),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    metrics=[tf.keras.metrics.RootMeanSquaredError()]
)

"""Membangun *Callbacks* jika tidak terdapat peningkatan terhadap `val_root_mean_squared_error` dengan *patience* 5 dan melatih model"""

# Early stopping untuk menghentikan pelatihan jika tidak ada peningkatan
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_root_mean_squared_error',
    patience=5,
    restore_best_weights=True
)

# Melatih model
history = model.fit(
    x=x_train,
    y=y_train,
    batch_size=64,
    epochs=100,
    validation_data=(x_val, y_val),
    callbacks=[early_stopping]
)

# Menampilkan grafik pelatihan
plt.figure(figsize=(12, 4))

# Mendapatkan epoch dengan RMSE validasi terendah (model terbaik)
best_epoch = np.argmin(history.history['val_root_mean_squared_error'])

plt.subplot(1, 2, 1)
plt.plot(history.history['root_mean_squared_error'], '-')
plt.plot(history.history['val_root_mean_squared_error'], '-')
plt.plot(best_epoch, history.history['val_root_mean_squared_error'][best_epoch], 'ro', markersize=8)
plt.title('Model RMSE')
plt.ylabel('RMSE')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation', 'Best Model'], loc='upper right')
plt.grid(True, linestyle='--', alpha=0.6)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], '-')
plt.plot(history.history['val_loss'], '-')
plt.plot(best_epoch, history.history['val_loss'][best_epoch], 'ro', markersize=8)
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation', 'Best Model'], loc='upper right')
plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.figtext(0.5, 0.01, f"Early stopping at epoch {best_epoch+1} (best model)",
            ha='center', fontsize=10, bbox=dict(facecolor='yellow', alpha=0.2))
plt.show()

"""Visualisai di atas menunjukkan performa model Collaborative Filtering dengan nilai *validation RMSE* yang menurun signifikan dari `0.2125` menjadi sekitar `0.1989` dan *validation loss* dari `0.0454` menjadi `0.0400`, mencapai performa optimal pada epoch ke-5 (ditandai dengan titik merah) dari total 10 epoch di mana early stopping diaktifkan, menunjukkan model yang efektif dengan generalisasi baik tanpa tanda-tanda overfitting.

### Testing
"""

# Membuat fungsi untuk mendapatkan rekomendasi film berdasarkan ID pengguna
def get_movie_recommendations(user_id, top_n=10):
    # Mendapatkan encoding pengguna
    user_encoded = cf_data_processed[cf_data_processed['user_id'] == user_id]['user_encoded'].iloc[0]

    # Mendapatkan film yang telah diberi rating oleh pengguna
    movies_rated_by_user = cf_data_processed[cf_data_processed['user_id'] == user_id]['movie_id'].tolist()

    # Mendapatkan semua encoding film
    all_movie_encodings = list(range(num_movies))

    # Membuat data input untuk prediksi
    user_movie_array = np.array([[user_encoded, movie_encoded] for movie_encoded in all_movie_encodings])

    # Mendapatkan prediksi rating
    ratings = model.predict(user_movie_array).flatten()

    # Membuat DataFrame dengan ID film, encoding, dan prediksi rating
    movie_ratings = pd.DataFrame({
        'movie_encoded': all_movie_encodings,
        'movie_id': [movie_encoded_to_movie[movie_encoded] for movie_encoded in all_movie_encodings],
        'predicted_rating': ratings
    })

    # Menghapus film yang sudah diberi rating oleh pengguna
    recommendations = movie_ratings[~movie_ratings['movie_id'].isin(movies_rated_by_user)]

    # Mengurutkan berdasarkan prediksi rating secara menurun
    recommendations = recommendations.sort_values('predicted_rating', ascending=False)

    # Menggabungkan dengan informasi film untuk mendapatkan judul
    recommendations_with_titles = recommendations.merge(
        movies[['movie_id', 'movie_title']],
        on='movie_id',
        how='left'
    )

    return recommendations_with_titles.head(top_n)

sample_user_id = 1
user_recommendations = get_movie_recommendations(sample_user_id, top_n=10)
print(f"Top 10 movie recommendations for User {sample_user_id}:")
for idx, row in user_recommendations.iterrows():
    print(f"{row['movie_title']} - Predicted rating: {row['predicted_rating'] * 5:.2f}/5.00")

sample_user_id = 21
user_recommendations = get_movie_recommendations(sample_user_id, top_n=10)
print(f"Top 10 movie recommendations for User {sample_user_id}:")
for idx, row in user_recommendations.iterrows():
    print(f"{row['movie_title']} - Predicted rating: {row['predicted_rating'] * 5:.2f}/5.00")

