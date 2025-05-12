# Proyek Kedua Movie Recommendation System

Disusun oleh: **Junianto Endra Kartika**

Proyek *Machine Learning Terapan* ini akan membangun sistem rekomendasi movie bagi pengguna.

## Project Overview

<p align="center">
  <img src="https://movielens.org/images/site/main-screen.png" />
</p>

<p align="center">Gambar 1. Cover</p>

Proyek ini mengembangkan sebuah sistem rekomendasi film yang canggih dengan memanfaatkan kekuatan algoritma pembelajaran mesin untuk memahami dan memprediksi preferensi pengguna. Sistem ini mengintegrasikan dua pendekatan utama: *Collaborative Filtering*, yang mengidentifikasi pengguna dengan selera serupa untuk merekomendasikan film berdasarkan penilaian kolektif, dan *Content-Based Filtering*, yang menganalisis karakteristik intrinsik film seperti genre, sutradara, dan aktor untuk menyarankan konten yang relevan. Melalui kombinasi sinergis kedua teknik ini, proyek ini bertujuan untuk menyajikan rekomendasi film yang sangat dipersonalisasi dan akurat, sehingga secara signifikan meningkatkan pengalaman pengguna, mendorong kepuasan, serta memperkuat loyalitas pelanggan seperti yang didukung oleh penelitian [1][2].

## Business Understanding

Bagian ini bertujuan untuk mengklarifikasi masalah yang mendasari pengembangan sistem rekomendasi film ini. Melalui pemahaman yang mendalam terhadap tantangan yang ada, kita dapat merumuskan tujuan yang jelas dan solusi yang terukur.

### Problem Statement

- Bagaimana cara mengatasi kelebihan informasi dan mempermudah pengguna menemukan film yang relevan di antara banyaknya pilihan yang tersedia?
- Mengapa rekomendasi film yang bersifat generik kurang memuaskan preferensi individual pengguna?
- Bagaimana platform penyedia film dapat memanfaatkan sistem rekomendasi yang lebih baik untuk meningkatkan engagement dan retensi penggunanya?

### Goals

- Meningkatkan Kemudahan Penemuan Film yang Relevan
- Meningkatkan Tingkat Personalisasi Rekomendasi Film
- Memaksimalkan Engagement dan Retensi Pengguna Melalui Rekomendasi yang Efektif

### Solution Statements

- Melakukan Eksplorasi dan Visualisasi Data Awal
- Implementasi Pendekatan *Content-Based Filtering*
- Implementasi Pendekatan *Collaborative Filtering*

## Data Understanding

Dataset yang digunakan dalam proyek ini adalah **MovieLens 100k Dataset** yang bersumber dari [GroupLens](https://grouplens.org/datasets/movielens/100k/).

### Deskripsi Dataset

Karena dataset memiliki format yang berbeda-beda, penelitian ini memerlukan `Data Loading` untuk menyiapkan data untuk menjadi data yang siap untuk dikembangkan. Berikut adalah data yang akan digunakan pada penelitian ini:

- User (Jumlah pengguna: 943 | 5 Kolom)
  - `user_id`: ID pengguna
  - `age`: umur pengguna
  - `sex`: jenis kelamin pengguna
  - `occupation`: pekerjaan pengguna
  - `zip_code`: kode alamat pengguna

- Genre (Jumlah genre: 19 | 2 Kolom)
  - `genre_name`: nama genre
  - `genre_id`: ID genre

- Movie (Jumlah film: 1682 | 24 Kolom)
  - `movie_id`: ID film
  - `movie_title`: judul film
  - `release_date`: tanggal terbit film
  - `video_release_date`: tanggal video film terbit
  - `IMDb_URL`: URL IMDb film
  - Sisa dari kolom `movie` adalah *One-hot encoded* genre

- Rating (Jumlah rating: 100000 / 100k | 4 Kolom)
  - `user_id`: ID user
  - `movie_id`: ID film
  - `rating`: rating yang diberikan user terhadap film
  - `unix_timestamps`: tanggal rating diberikan oleh user dalam bentuk *unix*

> Rating memiliki 2 data yaitu `ratings_base` dan `ratings_test`. Dimana jumlah rating sebelumnya merupakan gabungan dari 2 data tersebut.

Semua data yang digunakan tidak memiliki nilai yang kosong atau `null`

### Exploratory Data Analysis (EDA)

*Exploratory Data Analysis* (EDA) merupakan tahap awal analisis data untuk mengenali sifat-sifat utama, susunan, dan elemen penting dalam dataset sebelum analisis statistik atau prediksi yang lebih mendalam.

Berikut adalah tahapan EDA yang telah dilakukan:

#### Users

Eksplorasi data pada users akan memperlihatkan distribusi pemberian rating tiap user

<p align="center">
  <img src="https://raw.githubusercontent.com/J3ndra/dicoding_machine_learning_terapan/refs/heads/main/Recommendations%20System/Images/Users.png" />
</p>

<p align="center">Gambar 2. Distribusi rating tiap user</p>

Dengan rata-rata jumlah penilaian tiap user adalah `106` rating, dengan pemberian jumlah rating terbanyak pada user adalah `737` dan pemberian jumlah rating terendah pada user adalah `20`.

#### Genres

<p align="center">
  <img src="https://raw.githubusercontent.com/J3ndra/dicoding_machine_learning_terapan/refs/heads/main/Recommendations%20System/Images/Genres.png" />
</p>

<p align="center">Gambar 3. Jumlah film tiap genre</p>

Total genre adalah 19 genre dengan jumlah film terbanyak terdapat pada genre **Drama** sedangkan jumlah film terendah terdapat pada genre **Unknown**.

#### Ratings

<p align="center">
  <img src="https://raw.githubusercontent.com/J3ndra/dicoding_machine_learning_terapan/refs/heads/main/Recommendations%20System/Images/Ratings.png" />
</p>

<p align="center">Gambar 4. Distribusi nilai rating</p>

Gambar 4 menunjukkan bahwa nilai distribusi rating yang baik dengan nilai terendah adalah `1` dan nilai tertinggi adalah `5`, distribusi rating terbanyak terdapat pada rating `4` dengan rata-rata rating adalah `3.53`.

## Data Preprocessing && Data Preparation

### Data Preprocessing

#### Pembersihan Genre yang Tidak Diketahui

Karena genre `Unknown` hanya memiliki total 2 film, maka genre `Unknown` akan dihapus untuk menjaga keseimbangan data dengan cara:

1. Mencari dan meninjau apakah *movies* dengan *genre* `Unknown` memiliki genre `Unknown` saja atau memiliki genre yang lain

```py
# output
Number of movies with 'unknown' genre: 2

Details of unknown genre movies:
      movie_id          movie_title release_date
266        267              unknown          NaN
1372      1373  Good Morning (1971)   4-Feb-1971

Movies with 'unknown' genre that also have other genres: 0
Movies with ONLY 'unknown' genre: 2
```

2. Ternyata, terdapat *movies* dengan *title* `unknown` sehingga *movie* tersebut akan dihapus. Lalu, terdapat *movie* dengan *genre* `unknown` yang berjudul `Good Morning (1971)`. Tinjauan lanjutan akan mencari apakah *movie* tersebut memiliki *genre* yang lain. Karena tidak terdapat *genre* lain pada *movie* tersebut, maka *movie* tersebut akan dihapus.

> Setelah menghapus *genre* `unknown` dan *movie* yang memiliki genre `unknown`. Jumlah *movie* sebelumnya berjumlah `1682` menjadi `1680` dan jumlah *genre* yang sebelumnya `19` menjadi `18`.

#### Pengolahan Data untuk Content Based Filtering

Pada pendekatan *Content-Based Filtering*, data diolah melalui beberapa tahap penting untuk memungkinkan sistem merekomendasikan film berdasarkan karakteristik intrinsiknya:

1. **Ekstraksi Fitur Genre**: Data film yang berisi encoding one-hot untuk 18 genre digunakan sebagai basis utama. Setiap film memiliki representasi biner (0 atau 1) untuk setiap genre yang menandakan apakah film tersebut termasuk dalam kategori genre tersebut.

2. **Konversi Data Genre ke Format Teks**: Karena proyek ini akan menggunakan [**TF-IDF**](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html), kita akan menggabungkan semua *One-hot Encoded genre* yang terdapat pada data *movie* menjadi satu kolom bernama *genres* seperti `Animation Children's Comedy` atau `Action Advanture Thriller`.

|   | movie_id |       movie_title | Action | Adventure | Animation | Children's | Comedy | Crime | Documentary | Drama | ... | Horror | Musical | Mystery | Romance | Sci-Fi | Thriller | War | Western | release_year |                      genres |
|--:|---------:|------------------:|-------:|----------:|----------:|-----------:|-------:|------:|------------:|------:|----:|-------:|--------:|--------:|--------:|-------:|---------:|----:|--------:|-------------:|----------------------------:|
| 0 |        1 |  Toy Story (1995) |      0 |         0 |         1 |          1 |      1 |     0 |           0 |     0 | ... |      0 |       0 |       0 |       0 |      0 |        0 |   0 |       0 |         1995 | Animation Children's Comedy |
| 1 |        2 |  GoldenEye (1995) |      1 |         1 |         0 |          0 |      0 |     0 |           0 |     0 | ... |      0 |       0 |       0 |       0 |      0 |        1 |   0 |       0 |         1995 |   Action Adventure Thriller |
| 2 |        3 | Four Rooms (1995) |      0 |         0 |         0 |          0 |      0 |     0 |           0 |     0 | ... |      0 |       0 |       0 |       0 |      0 |        1 |   0 |       0 |         1995 |                    Thriller |
| 3 |        4 | Get Shorty (1995) |      1 |         0 |         0 |          0 |      1 |     0 |           0 |     1 | ... |      0 |       0 |       0 |       0 |      0 |        0 |   0 |       0 |         1995 |         Action Comedy Drama |
| 4 |        5 |    Copycat (1995) |      0 |         0 |         0 |          0 |      0 |     1 |           0 |     1 | ... |      0 |       0 |       0 |       0 |      0 |        1 |   0 |       0 |         1995 |        Crime Drama Thriller |

Tabel 1. Hasil pengolahan data untuk *Content Based Filtering*

#### Pengolahan Data untuk Collaborative Filtering

Pada pendekatan *Collaborative Filtering*, data diolah untuk memungkinkan model deep learning mempelajari pola interaksi antara pengguna dan film:

1. **Integrasi Data Pengguna dan Film**: Data rating digabungkan dengan informasi pengguna (seperti usia, jenis kelamin, pekerjaan) dan informasi film (judul) untuk memberikan konteks tambahan pada interaksi pengguna-film.

2. **Pembuatan Matriks Interaksi**: Matriks user-item dibuat yang merepresentasikan interaksi antara pengguna dan film, dengan nilai sel berupa rating yang diberikan pengguna pada film tertentu.

3. **Encoding ID**: Untuk efisiensi komputasi, ID pengguna dan film yang bersifat kategorikal dikonversi menjadi indeks numerik berurutan (0 sampai n-1). Pemetaan antara ID asli dan ID terenkode disimpan untuk memudahkan interpretasi hasil.

4. **Normalisasi Rating**: Nilai rating dinormalisasi ke skala 0-1 dengan membagi setiap nilai rating dengan 5 (nilai maksimum), yang memungkinkan model untuk bekerja dengan rentang nilai yang lebih seragam.

|   | user_id | movie_id | rating | unix_timestamp | user_encoded | movie_encoded |
|--:|--------:|---------:|-------:|---------------:|-------------:|--------------:|
| 0 |       1 |        1 |    1.0 |      874965758 |            0 |             0 |
| 1 |       1 |        2 |    0.6 |      876893171 |            0 |             1 |
| 2 |       1 |        3 |    0.8 |      878542960 |            0 |             2 |
| 3 |       1 |        4 |    0.6 |      876893119 |            0 |             3 |
| 4 |       1 |        5 |    0.6 |      889751712 |            0 |             4 |

Tabel 2. Hasil pengolahan data untuk `Collaborative Filtering`

### Data Preparation

Dalam menyiapkan data agar dapat diolah oleh model *Content-Based Filtering* dan juga *Collaborative Filtering* perlu dilakukan beberapa tahapan yaitu

- **Pengecekan missing values**
  
  Tidak terdapat missing values pada kedua data yang telah diolah, menandakan tidak perlu adanya penanganan missing values

- **Content-Based data preparation**

  Langkah awal yang ditempuh pada data *Content-Based Filtering* adalah mentransformasikan data genre film menjadi representasi vektor numerik. Proses ini dilakukan dengan memanfaatkan [*TfidfVectorizer*](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html) untuk mengukur frekuensi dan memberikan bobot pada setiap genre dalam film, yang kemudian diaplikasikan (fit) dan diubah (transform) menjadi sebuah matriks **TF-IDF** menggunakan [*Cosine Similarity*](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html).

  $$
  \text{Cosine Similarity}(A, B) = \frac{A \cdot B}{\|A\| \|B\|}
  $$

  <p align="center">Rumus 1. Rumus Consine Similarity</p>

  |                      | war | drama | western |  fi | mystery | action | children | horror | crime |   comedy |
  |---------------------:|----:|------:|--------:|----:|--------:|-------:|---------:|-------:|------:|---------:|
  |          movie_title |     |       |         |     |         |        |          |        |       |          |
  |        Grease (1978) | 0.0 |   0.0 |     0.0 | 0.0 |     0.0 |    0.0 | 0.000000 |    0.0 |   0.0 | 0.385700 |
  | Dirty Dancing (1987) | 0.0 |   0.0 |     0.0 | 0.0 |     0.0 |    0.0 | 0.000000 |    0.0 |   0.0 | 0.000000 |
  |    Home Alone (1990) | 0.0 |   0.0 |     0.0 | 0.0 |     0.0 |    0.0 | 0.854178 |    0.0 |   0.0 | 0.519981 |
  |          Nell (1994) | 0.0 |   1.0 |     0.0 | 0.0 |     0.0 |    0.0 | 0.000000 |    0.0 |   0.0 | 0.000000 |
  |   Mighty, The (1998) | 0.0 |   1.0 |     0.0 | 0.0 |     0.0 |    0.0 | 0.000000 |    0.0 |   0.0 | 0.000000 |

  Tabel 3. Hasil *Cosine Similarity*

  Tabel 3 menunjukkan kemiripan genre antar film. Nilai mendekati 1 berarti film sangat mirip dengan genre kolom, sedangkan mendekati 0 berarti tidak mirip dimana:

  - Grease: Cukup mirip dengan comedy.
  - Dirty Dancing: Tidak mirip dengan genre yang ada.
  - Home Alone: Sangat mirip dengan children dan cukup mirip dengan comedy.
  - Nell & Mighty: Identik dengan genre drama.

- **Collaborative data preparation**

  Pada *Collaborative filtering* data, dilakukan pembagian data menjadi train 80% dan validasi 20% menggunakan [*train_test_split*](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) dengan jumlah data pada data latih adalah `79992` dan data uji `19998`.

## Modeling

Dalam proyek ini, tahapan pemodelan memanfaatkan dua pendekatan algoritmik utama: Jaringan Neural (*Neural Network*) dan Kesamaan Kosinus (*Cosine Similarity*). Jaringan Neural akan diimplementasikan dalam sistem rekomendasi dengan paradigma *Collaborative Filtering*, sementara Kesamaan Kosinus akan diterapkan pada model *Content-Based Filtering*.

### Content-Based Filtering

Dalam proses pembentukan model *Content-Based Filtering*, salah satu langkah penting adalah membangun mekanisme yang mampu menerima sebuah judul film dan kemudian merekomendasikan film-film lain yang dianggap mirip berdasarkan analisis konten film tersebut, sekaligus mengukur seberapa relevan rekomendasi tersebut berdasarkan kategori genre yang ada. 

*Content-Based Filtering* memiliki beberapa kelebihan dan kekurang seperti:

- Kelebihan:
  - Rekomendasi sesuai selera
  - Alasan rekomendasi jelas

- Kekurangan:
  - Membutuhkan informasi genre yang lengkap
  - Sulit menemukan film baru di luar selera

Rekomendasi 10 film berdasarkan *Content Based Filtering* tersedia di tabel 5, dengan film sumber rekomendasi tertera di tabel 4.

| movie_id | title            | genres                        |
|----------|------------------|-------------------------------|
| 0        | Toy Story (1995) | Animation, Comedy, Children's |

Tabel 4. Data uji coba

| title                                                                   | genre     |
|-------------------------------------------------------------------------|-----------|
| Aladdin and the King of Thieves (1996)                                  | Animation |
| Aristocats, The (1970)                                                  | Animation |
| Pinocchio (1940)                                                        | Animation |
| Sword in the Stone, The (1963)                                          | Animation |
| Fox and the Hound, The (1981)                                           | Animation |
| Winnie the Pooh and the Blustery Day (1968)                              | Animation |
| Balto (1995)                                                            | Animation |
| Oliver & Company (1988)                                                 | Animation |
| Swan Princess, The (1994)                                               | Animation |
| Land Before Time III: The Time of the Great Giving (1995) (V)            | Animation |

Tabel 5. Hasil rekomendasi *Content-Based Filtering*

### Collaborative Filtering

Pada model Collaborative Filtering, informasi interaksi antara pengguna dan film sangat dibutuhkan agar model dapat bekerja dengan baik.

Kelas model yang akan dibagun akan menggunakan [*Keras Model Class*](https://keras.io/api/models/model/) yang terinspirasi dari [*Collaborative Filtering MovieLens* dari Keras](https://keras.io/examples/structured_data/collaborative_filtering_movielens/). Model ini akan memanfaatkan embedding layers untuk merepresentasikan pengguna dan film dalam ruang vektor laten, sehingga memungkinkan untuk mempelajari preferensi pengguna dan karakteristik film secara tersembunyi."

Sebelum model dilatih, proyek ini akan menggunakan [*Callbacks*](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/Callback) [*EarlyStopping*](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/EarlyStopping) jika tidak terdapat peningkatan terhadap `val_root_mean_squared_error` dengan *patience* 5 dan melatih model

*Collaborative Filtering* memiliki beberapa kelebihan dan kekurang seperti:

- Kelebihan:
  - Dapat merekomendasikan film tanpa mengetahui detail film
  - Dapat menemukan film yang cocok sesuai pengguna berdasarkan selera pengguna lain
  - Sistem rekomendasi akan lebih baik dengan data yang lebih banyak

- Kekurangan:
  - Sulit merekomendasikan film baru atau ke pengguna baru yang belum ada ratingnya.
  - Jika banyak pengguna hanya memberi rating sedikit film, datanya jadi kurang lengkap.
  - Kurang merekomendasikan film yang unik atau *niche* jika tidak terdapat banyak pengguna yang menyukai film tersebut.

Pengujian dilakukan menggunakan 2 user, yaitu rekomendasi user dengan ID "1" yang tertera pada tabel 6 dan rekomendasi user dengan ID "21" yang tertera pada tabel 7.

#### Pengujian pada user dengan ID "1"

|    | Title                                      | Predicted Rating |
|----|--------------------------------------------|------------------|
| 1  | Schindler's List (1993)                    | 4.44/5.00        |
| 2  | Titanic (1997)                             | 4.35/5.00        |
| 3  | Close Shave, A (1995)                      | 4.20/5.00        |
| 4  | As Good As It Gets (1997)                 | 4.20/5.00        |
| 5  | Apt Pupil (1998)                           | 4.18/5.00        |
| 6  | L.A. Confidential (1997)                   | 4.16/5.00        |
| 7  | To Kill a Mockingbird (1962)               | 4.12/5.00        |
| 8  | Secrets & Lies (1996)                      | 4.12/5.00        |
| 9  | Mrs. Brown (Her Majesty, Mrs. Brown) (1997) | 4.08/5.00        |
| 10 | Casablanca (1942)                          | 4.08/5.00        |

Tabel 6. hasil pengujian pada user dengan ID "1"

#### Pengujian pada user dengan ID "21"

|    | Title                                                              | Predicted Rating |
|----|--------------------------------------------------------------------|------------------|
| 1  | Shawshank Redemption, The (1994)                                   | 3.83/5.00        |
| 2  | Usual Suspects, The (1995)                                         | 3.78/5.00        |
| 3  | One Flew Over the Cuckoo's Nest (1975)                             | 3.74/5.00        |
| 4  | Dr. Strangelove or: How I Learned to Stop Worrying and Love the Bomb (1963) | 3.65/5.00        |
| 5  | Blade Runner (1982)                                                | 3.64/5.00        |
| 6  | Casablanca (1942)                                                  | 3.64/5.00        |
| 7  | Lawrence of Arabia (1962)                                          | 3.63/5.00        |
| 8  | Boot, Das (1981)                                                   | 3.58/5.00        |
| 9  | Fugitive, The (1993)                                               | 3.57/5.00        |
| 10 | Schindler's List (1993)                                            | 3.56/5.00        |

Tabel 7. hasil pengujian pada user dengan ID "21"

## Evaluasi

Dalam pendekatan *Content-Based Filtering*, metrik evaluasi utama yang digunakan adalah *Cosine Similarity*. Nilai *Cosine Similarity* yang lebih tinggi mengindikasikan tingkat kemiripan yang lebih besar antara dua item. Rumus perhitungan *Cosine Similarity* dapat dilihat pada rumus 1.

### Content-Based Filtering

Pada evaluasi content based filtering menggunakan metrik precision content based filtering untuk menghitung precision model sistem yang telah dibuat sebelumnya. Rumus perhitungan precision dapat dilihat pada rumus 3 dengan pengujian dilakukan menggunakan 2 film, yaitu "Toy Story (1995)" dan "You So Crazy (1994)".

$$
\text{Recall at } K = \frac{\text{Number of relevant items in } K}{\text{Total number of relevant items}}
$$

<p align="center">Rumus 3. Rumus precision</p>

#### Pengujian pada film "Toy Story (1995)"

> Target genres: Comedy, Animation, Children's

|     | title                                                          | genre     | similarity |
| :-- | :------------------------------------------------------------- | :-------- | :---------- |
| 1   | Aladdin and the King of Thieves (1996)                         | Animation | 1.0002      |
| 2   | Aristocats, The (1970)                                         | Animation | 0.9373      |
| 3   | Pinocchio (1940)                                               | Animation | 0.9374      |
| 4   | Sword in the Stone, The (1963)                                 | Animation | 0.9375      |
| 5   | Fox and the Hound, The (1981)                                  | Animation | 0.9376      |
| 6   | Winnie the Pooh and the Blustery Day (1968)                     | Animation | 0.9377      |
| 7   | Balto (1995)                                                   | Animation | 0.9378      |
| 8   | Oliver & Company (1988)                                        | Animation | 0.9379      |
| 9   | Swan Princess, The (1994)                                      | Animation | 0.93710     |
| 10  | Land Before Time III: The Time of the Great Giving (1995) (V) | Animation | 0.937       |

Tabel 8. Hasil pengujian pada film "Toy Story (1995)"

Tabel 8 menunjukkan bahwa film **Toy Story (1995)** memiliki 3 *genre* yaitu `Comedy, Animation, Children's` dimana film **Aladdin and the King of Thieves (1996)** memiliki 3 *genre* yang sama sehingga memiliki *score similarity* mencapai `1.0`. Sedangkan film yang lain hanya memiliki 2 *genre* yang sama, maka dari itu *cosine similarity* hanya memiliki score `0.937`.

#### Pengujian pada film "You So Crazy (1994)"

> Target genres: Comedy

|     | title                                                        | genre  | similarity |
| :-- | :----------------------------------------------------------- | :----- | :---------- |
| 1   | Birdcage, The (1996)                                       | Comedy | 1.0       |
| 2   | Brothers McMullen, The (1995)                              | Comedy | 1.0        |
| 3   | To Wong Foo, Thanks for Everything! Julie Newmar (1995)    | Comedy | 1.0        |
| 4   | Billy Madison (1995)                                       | Comedy | 1.0        |
| 5   | Clerks (1994)                                              | Comedy | 1.0        |
| 6   | Ace Ventura: Pet Detective (1994)                          | Comedy | 1.0        |
| 7   | Ref, The (1994)                                            | Comedy | 1.0        |
| 8   | Theodore Rex (1995)                                        | Comedy | 1.0        |
| 9   | Sgt. Bilko (1996)                                          | Comedy | 1.0       |
| 10  | Kids in the Hall: Brain Candy (1996)                      | Comedy | 1.0         |

Tabel 9. hasil pengujian pada film "You So Crazy (1994)"

Tabel 9 menunjukkan bahwa semua film yang direkomendasikan memiliki *score similarity* **1.0**, karena film yang ingin direkomendasikan hanya memiliki 1 genre yang `Comedy`.

### Collaborative Filtering

Dalam pendekatan *Collaborative Filtering*, metrik evaluasi utama yang digunakan adalah *Root Mean Square Error* (**RMSE**). Nilai **RMSE** yang lebih rendah mengindikasikan kinerja model yang lebih akurat dalam memprediksi rating pengguna. Rumus perhitungan **RMSE** dapat dilihat di rumus 4.

$$
RMSE = \sqrt{\frac{\sum_{i=1}^{n} (\hat{y}_i - y_i)^2}{n}}
$$

<p align="center">Rumus 4. Rumus RMSE</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/J3ndra/dicoding_machine_learning_terapan/refs/heads/main/Recommendations%20System/Images/rmse_epoch.png" />
</p>

<p align="center">Gambar 5. Visualisasi epochs RMSE</p>

Gambar 5. menunjukkan performa model Collaborative Filtering dengan nilai *validation RMSE* yang menurun signifikan dari `0.2125` menjadi sekitar `0.1989` dan *validation loss* dari `0.0454` menjadi `0.0400`, mencapai performa optimal pada epoch ke-6 (ditandai dengan titik merah) dari total 11 epoch di mana early stopping diaktifkan, menunjukkan model yang efektif dengan generalisasi baik tanpa tanda-tanda overfitting.


## Kesimpulan

Berdasarkan evaluasi sistem rekomendasi yang dikembangkan, pendekatan *Content-Based Filtering* dan *Collaborative Filtering* berhasil digunakan untuk memberikan rekomendasi film yang relevan kepada pengguna. Meskipun *Content-Based Filtering* efektif dalam merekomendasikan film berdasarkan kesamaan konten, pendekatan *Collaborative Filtering* menunjukkan kinerja yang lebih baik dalam mempersonalisasi rekomendasi, dengan hasil prediksi rating yang cukup akurat berdasarkan metrik **RMSE**. 

Secara keseluruhan, sistem yang dibangun mampu menjawab permasalahan bisnis dengan membantu pengguna menemukan film yang sesuai dengan preferensi mereka secara lebih efisien dan akurat.

## Referensi

[1] Liu, Y., Xu, Y., & Zhou, S. (2024). Enhancing User Experience through Machine Learning-Based Personalized Recommendation Systems: Behavior Data-Driven UI Design. Applied and Computational Engineering, 112(1), 42–46. https://doi.org/10.54254/2755-2721/2024.17905

[2] Madavi, N. P., Tekam, N. J., & Khirekar, N. P. D. (2024). Movie recommendation system. International Journal of Advanced Research in Science Communication and Technology, 320–324. https://doi.org/10.48175/ijarsct-17453