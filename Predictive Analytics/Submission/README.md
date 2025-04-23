# Laporan Proyek Predictive Analytics - Junianto Endra Kartika

<p align="center">
  <img src="https://github.com/J3ndra/dicoding_machine_learning_terapan/blob/main/Predictive%20Analytics/Images/image_1.png?raw=true" />
</p>

<p align="center">Gambar 1. Cover</p>

## Project Overview

Pertanian merupakan sektor fundamental yang menopang kehidupan manusia dan perekonomian global. Pemilihan jenis tanaman yang tepat untuk ditanam pada suatu lahan merupakan keputusan krusial yang secara langsung memengaruhi produktivitas hasil panen, efisiensi penggunaan sumber daya, dan keberlanjutan praktik pertanian. Secara tradisional, keputusan ini seringkali didasarkan pada pengalaman turun-temurun petani atau intuisi, yang mungkin tidak selalu optimal mengingat kompleksitas dan variabilitas faktor lingkungan seperti kondisi tanah (kandungan N, P, K, pH), iklim (suhu, kelembaban, curah hujan), dan faktor lainnya. Ketidakpastian akibat perubahan iklim semakin menambah tantangan dalam menentukan tanaman yang paling sesuai untuk kondisi spesifik suatu area.

Di sinilah peran teknologi, khususnya Machine Learning (ML), menjadi sangat relevan dan penting. Proyek ini berfokus pada pengembangan sistem rekomendasi tanaman (crop recommendation) menggunakan algoritma Machine Learning [1][2][3].

## Business Understanding

Bagian ini merangkum masalah bisnis, tujuan proyek, dan pendekatan solusi untuk sistem rekomendasi tanaman berbasis Machine Learning.

### Problem Statements

- Kesulitan Optimasi Tanaman: Petani sulit menentukan tanaman paling optimal karena kompleksitas data kondisi tanah (N, P, K, pH) dan iklim (suhu, kelembaban, curah hujan) [1].
- Risiko Inefisiensi: Keputusan tanam yang kurang tepat berisiko menurunkan hasil panen dan menyebabkan pemborosan sumber daya (pupuk, air) [1].

### Goals

- Pengembangan Model Prediktif: Membuat model Machine Learning yang mampu memprediksi dan merekomendasikan tanaman terbaik berdasarkan analisis data tanah dan iklim.
- Peningkatan Efisiensi Pertanian: Menyediakan alat bantu rekomendasi untuk membantu petani memaksimalkan panen dan mengoptimalkan penggunaan sumber daya.

### Solution Approach

Pendekatan solusi untuk mencapai tujuan tersebut adalah dengan menguji dan membandingkan performa beberapa algoritma klasifikasi Machine Learning yang umum digunakan untuk tugas semacam ini:

- ***Random Forest Classifier*** (RF): Algoritma ensemble yang menggabungkan banyak decision tree untuk akurasi dan stabilitas tinggi, serta mampu menangani hubungan non-linear.
- ***K-Nearest Neighbors*** (KNN): Algoritma berbasis jarak yang mengklasifikasikan data baru berdasarkan mayoritas *tetangga* terdekatnya di ruang fitur.
- ***Logistic Regression*** (LR): Meskipun namanya regresi, ini adalah model klasifikasi linear yang memprediksi probabilitas kelas.

Ketiga model ini akan dilatih menggunakan data yang ada, dioptimalkan menggunakan teknik seperti ***GridSearchCV*** untuk menemukan hyperparameter terbaik , dan dievaluasi menggunakan ***Confusion Matrix*** untuk melihat hasil akurasi pelatihan.

## Data Understanding

Dataset yang digunakan dalam proyek ini adalah **Crop Recommendation Dataset** yang bersumber dari [Kaggle](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset).

### Informasi dataset:

- Dataset memiliki format `CSV`
- Tidak terdapat *Missing Values* dan *Error Values*
- Dataset terdiri dari 2200 sample dengan 7 fitur numerik serta 1 fitur kategorikal

### Informasi masing-masing fitur:

- **N**: Rasio kandungan Nitrogen di dalam tanah.
- **P**: Rasio kandungan Fosfor (Phosphorous) di dalam tanah.
- **K**: Rasio kandungan Kalium (Potassium) di dalam tanah.
- **temperature**: Suhu rata-rata lingkungan dalam derajat Celsius (°C).
- **humidity**: Kelembaban relatif udara dalam persentase (%).
- **ph**: Nilai pH (tingkat keasaman atau kebasaan) tanah.
- **rainfall**: Jumlah curah hujan rata-rata dalam milimeter (mm).
- **label**: Jenis tanaman yang direkomendasikan atau cocok untuk ditanam pada kondisi tersebut (Variabel Target).

### Exploratory Data Analysis (EDA)

Exploratory Data Analysis (EDA) merupakan tahap awal analisis data untuk mengenali sifat-sifat utama, susunan, dan elemen penting dalam dataset sebelum analisis statistik atau prediksi yang lebih mendalam.

Berikut adalah tahapan EDA yang telah saya lakukan:

#### Hubungan antara label dengan fitur

<p align="center">
  <img src="https://i.ibb.co.com/d0StQBcN/image.png" />
</p>

<p align="center">Gambar 2. Hubungan antara label dengan fitur</p>

Visualisasi pada Gambar 2 ini sangat berguna dalam melihat potensi separabilitas kelas sebelum membangun model klasifikasi serta memberikan gambaran awal mengenai kompleksitas data yang akan dihadapi oleh model.

Beberapa pemahan data yang dapat diambil dari visualisasi tersebut adalah:

- Pemisahan *cluster* yang cukup jelas pada beberapa fitur, seperti fitur `N` dengan fitur `P`, fitur `temperature` dengan fitur `humidity`, serta fitur `rainfall` dengan fitur `K` yang menunjukkan bahwa beberapa jenis tanaman atau *kelas* memiliki rentang nilai yang khas dan berbeda.
- Fitur `K` memiliki nilai prediktif yang kuat, khususnya untuk memisahkan jenis tanaman atau *kelas* yang membutuhkan *Kalium* `K` tinggi. Perbedaan nilai prediktif yang kuat ini dapat menimbulkan *miss informasi* ketika dilakukan pengecekan *outliers*.
- Fitur `rainfall` menunjukkan variasi yang cukup luas antar tanaman, dimana beberapa jenis tanaman atau *kelas* tampaknya memerlukan curah hujan yang lebih tinggi dibanding yang lain.

#### Pengecekan deskripsi dataset pada fitur numerikal

Saya memisahkan fitur numerikal dan kategorikal, kemudian fitur numerikal di analisa menggunakan fungsi `describe()`.

|       |      N      |      P      |      K      | temperature |   humidity  |      ph     |   rainfall  |
|------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|
| count | 2200.000000 | 2200.000000 | 2200.000000 | 2200.000000 | 2200.000000 | 2200.000000 | 2200.000000 |
|  mean |   50.551818 |   53.362727 |   48.149091 |   25.616244 |   71.481779 |    6.469480 |  103.463655 |
|   std |   36.917334 |   32.985883 |   50.647931 |    5.063749 |   22.263812 |    0.773938 |   54.958389 |
|   min |    0.000000 |    5.000000 |    5.000000 |    8.825675 |   14.258040 |    3.504752 |   20.211267 |
|   25% |   21.000000 |   28.000000 |   20.000000 |   22.769375 |   60.261953 |    5.971693 |   64.551686 |
|   50% |   37.000000 |   51.000000 |   32.000000 |   25.598693 |   80.473146 |    6.425045 |   94.867624 |
|   75% |   84.250000 |   68.000000 |   49.000000 |   28.561654 |   89.948771 |    6.923643 |  124.267508 |
|   max |  140.000000 |  145.000000 |  205.000000 |   43.675493 |   99.981876 |    9.935091 |  298.560117 |

<p align="center">Tabel 1. Deskripsi dataset</p>

Dari tabel 1 terlihat bahwa fitur `N` memiliki nilai minimum 0, namun tidak dapat diasumsikan secara langsung jika nilai tersebut adalah nilai error karena kemungkinan memang terdapat *crop* yang mengharuskan nilai pada fitur `N` adalah 0.

#### Distribusi pada label

<p align="center">
  <img src="https://raw.githubusercontent.com/J3ndra/dicoding_machine_learning_terapan/refs/heads/main/Predictive%20Analytics/Images/image_2.png" />
</p>

<p align="center">Gambar 3. Distribusi label</p>

Gambar 3 memperlihatkan kita bahwa distribusi label pada semua label seimbang yaitu di `100 data`.

#### Univariate Analysis

Analisis univariat adalah jenis analisis statistik yang fokus pada satu variabel saja dalam satu waktu.

<p align="center">
  <img src="https://raw.githubusercontent.com/J3ndra/dicoding_machine_learning_terapan/refs/heads/main/Predictive%20Analytics/Images/image_5.png" />
</p>

<p align="center">Gambar 4. Visualisasi univariate analysis</p>

Berikut adalah hasil analisa gambar 4:

- N (**Nitrogen**): Distribusi nitogres condong ke kiri dengan mayoritas nilai berada di bawah 50
- P (**Fosforus**): Distribusi fosforus menyebar lebih merata dengan puncak di nilai sekitar 60 dan terdapat lonjakan pada nilai diatas 120
- K (**Potasium**): Distribusi potasium condong ke kiri dengan mayoritas nilai berada di bawah 50 dan terdapat nilai yang tinggi yaitu pada nilai sekitar 200
- *Temperature* (**Suhu**): Distribusi suhu simetris dan mendekati normal dengan puncak di sekitar 25 derajat celcius.
- *Humidity* (**Kelembaban**): Distribusi kelembaban condong ke kanan dengan sebagian besar data beradara di antara 60% hingga 90%.
- pH (**Keasaman Tanah**): Distribusi pH simetris dan mendekati normal dengan puncak di sekitar pH 6 dan pH 7.
- *Rainfall* (**Curah Hujan**): Distrubsi curah hujan condong ke kiri dengan mayoritas nilai berada di rentang 50mm hingan 150mm

#### Multivariate Analysis

Analisis multivariat adalah jenis analisis statistik yang melibatkan lebih dari satu variabel secara simultan.

<p align="center">
  <img src="https://raw.githubusercontent.com/J3ndra/dicoding_machine_learning_terapan/refs/heads/main/Predictive%20Analytics/Images/image_6.png" />
</p>

<p align="center">Gambar 5. Visualisasi multivariate analysis</p>

Gambar 5 menunjukkan nilai korelasi matriks diatas, fitur **K (Potasium)** dan **P (Fosforus)** mendapatkan nilai korelasi terbesar yaitu di angka `0.74`.

## Data Preparation

Pada tahap *Data Preparation*, pengecekan outliers akan dilakukan terlebih dahulu untuk dipastikan data sudah bersih dan terdistribusi dengan baik. Lalu data diubah menjadi format yang tepat untuk proses pemodelan. Proyek ini mengimplementasikan tiga tahapan penting, yaitu pemisahan data (Split Data) dan normalisasi (*Normalization*) dan encoder (*Label Encoder*).

### Pengecekan outliers

<p align="center">
  <img src="https://raw.githubusercontent.com/J3ndra/dicoding_machine_learning_terapan/refs/heads/main/Predictive%20Analytics/Images/image_3.png" />
</p>

<p align="center">Gambar 6. Visualisasi pengecekan outliers</p>

Menurut gambar 6, terdapat outliers pada data kita. Outlier merupakan nilai yang jauh berbeda dari mayoritas data dalam dataset. Penghapusan outlier dapat meningkatkan mutu analisis dan model prediksi. Untuk mengidentifikasi dan menangani outlier, visualisasi dengan library ***Seaborn*** digunakan, diikuti dengan penerapan teknik **IQR** pada data.

|  Sebelum  |  Sesudah  |
|:---------:|:---------:|
| (2200, 8) | (1846, 8) |

<p>Tabel 2. Output hasil pengurangan outliers</p>



> Namun, saat dilakukan pengecekan ulang pada distribusi label pada data (Gambar 7). Terdapat label yang hilang, sehingga dapat diasumsikan bahwa terdapat jenis tanaman yang memang berbeda cara penanganan nya dari jenis tanaman yang lain seperti yang telah dijelaskan pada bagian **Data Understanding**.

<p align="center">
  <img src="https://i.ibb.co.com/FLxPVfgS/image.png" />
</p>

<p align="center">Gambar 7. Distribusi label setelah penghapusan outliers</p>

### Split Data

Saat membagi data, fitur-fitur dipisahkan menjadi variabel independen (x) dan variabel target atau label (y). Pembagian data latih sebesar 80% dan data uji sebesar 20% dilakukan menggunakan fungsi [***TrainTestSplit***](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) dari library [***Sklearn***](https://scikit-learn.org/), seperti yang ditunjukkan pada tabel 3.

| Fitur |    Train   |    Test   |
|:-----:|:----------:|:---------:|
|   x   | (1760, 7) | (440, 7) |
|   y   |   (1760)   |   (440)   |

<p>Tabel 3. Hasil pemisahan data</p>

### Normalisasi

Normalisasi bertujuan untuk menciptakan distribusi data yang lebih baik, yang pada gilirannya dapat mengoptimalkan performa model dan akurasi prediksi. Dalam proyek ini, teknik [***MinMaxScaler***](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html) dari library [***Sklearn***](https://scikit-learn.org/) diterapkan. [***MinMaxScaler***](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html) menskalakan seluruh nilai fitur dalam dataset ke dalam rentang antara `0` dan `1`.

### Label Encoder

[***LabelEncoder***](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html) adalah sebuah teknik yang digunakan untuk mengubah label-label kategorikal (misalnya, teks) menjadi nilai numerik. Hal ini diperlukan karena sebagian besar algoritma machine learning bekerja dengan data numerik. Contoh pada data ***Crop Recommendation***, `rice`, `orange`, dan `apple`, [***LabelEncoder***](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html) akan mengubahnya menjadi `0`, `1`, dan `2` secara berurutan.

## Modeling

Tahap ini adalah pembentukan model machine learning dengan menerapkan algoritma pada data yang sudah diproses. Model dilatih menggunakan data latih untuk mengenali pola dan korelasi antara fitur dan target (label) prediksi.

### Algoritma

Algoritma yang akan dibangun adalah *K-Nearest Neighbour*, *Random Forest*, dan *Logistic Regression*.

#### K-Nearest Neighbour

KNN adalah algoritma klasifikasi (dan regresi) yang bekerja dengan cara mencari K titik data terdekat (tetangga terdekat) dalam ruang fitur terhadap titik data baru yang ingin diklasifikasikan. Kelas dari titik data baru kemudian ditentukan oleh mayoritas kelas di antara K tetangga terdekatnya.

- Kelebihan:

    - Sederhana dan mudah diimplementasikan.
    - Tidak memerlukan asumsi tentang distribusi data.
    - Berguna untuk data dengan batas keputusan yang non-linear.
    - Dapat digunakan untuk klasifikasi dan regresi.

- Kekurangan:

    - Sensitif terhadap skala fitur.
    - Mahal secara komputasi untuk dataset yang besar karena perlu menghitung jarak ke semua titik data.
    - Kinerja sangat bergantung pada pemilihan nilai K.
    - Rentan terhadap data yang tidak seimbang.

Parameter yang Anda gunakan:

- `n_neighbors`: Menentukan jumlah tetangga terdekat yang akan dipertimbangkan untuk klasifikasi.

- `metric`: Menentukan metrik jarak yang digunakan untuk mengukur kedekatan antar titik data.

- `weights`: Menetapkan bobot yang sama untuk semua tetangga dalam proses pengambilan keputusan.

#### Random Forest (RF)

Random Forest adalah algoritma ensemble learning yang membangun banyak pohon keputusan (decision tree) selama proses pelatihan. Untuk melakukan klasifikasi, Random Forest akan mengambil hasil prediksi dari setiap pohon dan memilih kelas yang paling banyak diprediksi (modus).

- Kelebihan:

    - Cenderung memberikan akurasi yang tinggi.
    - Robust terhadap outlier dan fitur yang tidak relevan.
    - Dapat menangani data dengan dimensi tinggi.
    - Memberikan ukuran pentingnya fitur.
    - Tidak rentan terhadap overfitting seperti single decision tree.

- Kekurangan:

    - Lebih sulit diinterpretasikan dibandingkan dengan single decision tree.
    - Membutuhkan waktu pelatihan yang lebih lama, terutama untuk jumlah pohon yang besar.

Parameter yang digunakan

- `max_depth`: Menentukan kedalaman maksimum dari setiap pohon keputusan. Ini mengontrol kompleksitas setiap pohon dan membantu mencegah overfitting.
- `min_samples_split`: Menentukan jumlah minimum sampel yang dibutuhkan untuk membagi node internal dalam sebuah pohon.
- `n_estimators`: Menentukan jumlah pohon keputusan yang akan dibangun dalam random forest.

#### Logistic Regression (LR)

Logistic Regression adalah algoritma klasifikasi linear yang digunakan untuk memprediksi probabilitas hasil biner (misalnya, 0 atau 1, ya atau tidak). Meskipun namanya mengandung "regresi", algoritma ini digunakan untuk klasifikasi dengan memodelkan probabilitas suatu kejadian menggunakan fungsi logistik (sigmoid).

- Kelebihan:

    - Mudah diinterpretasikan.
    - Efisien secara komputasi, terutama untuk dataset yang besar.
    - Memberikan probabilitas untuk setiap kelas.
    - Bekerja dengan baik untuk masalah klasifikasi linear.

- Kekurangan:

    - Hanya efektif untuk masalah yang dapat dipisahkan secara linear.
    - Sensitif terhadap outlier.
    - Mungkin kurang baik kinerjanya untuk masalah dengan batas keputusan yang kompleks.

Parameter yang digunakan:

- `C`: Merupakan parameter regularization (inverse of regularization strength). Nilai yang lebih kecil menunjukkan regularization yang lebih kuat, yang dapat membantu mencegah overfitting dengan memberikan penalti pada koefisien yang besar.
- `solver`: Menentukan algoritma optimasi yang digunakan untuk menemukan parameter model.

### GridSearch

Untuk mendapatkan parameter model terbaik, *Grid Search* secara otomatis mencoba semua kombinasi parameter yang kita tentukan. Hal ini membantu kita menemukan setelan parameter optimal untuk model kita tanpa mencoba satu per satu secara manual.

|        Model       | Best Score | Best Parameters                                                 |
|:------------------:|:----------:|-----------------------------------------------------------------|
|    RandomForest    |    0.995   | {'max_depth': 10, 'min_samples_split': 2, 'n_estimators': 150}  |
|         KNN        |   0.9836   | {'metric': 'manhattan', 'n_neighbors': 5, 'weights': 'uniform'} |
| LogisticRegression |   0.9609   | {'C': 10, 'solver': 'liblinear'}                                |

<p>Tabel 4. Hasil Grid Search</p>

Dari hasil Grid Search pada tabel 4, terlihat bahwa algoritma ***Random Forest*** memberikan skor terbaik yaitu `0.995` dengan kombinasi parameter `{'max_depth': 10, 'min_samples_split': 2, 'n_estimators': 150}`.

## Evaluation

Evaluasi dilakukan untuk mengetahui sejauh mana model yang dibangun dapat memenuhi kebutuhan bisnis yang telah dirumuskan pada tahap Business Understanding. Secara khusus, evaluasi ini bertujuan menjawab tiga hal: 

1. apakah model mampu memberikan rekomendasi tanaman secara akurat,
2. apakah pendekatan solusi berdampak pada permasalahan yang ada,
3. apakah tujuan proyek telah tercapai.

### Accuracy

Akurasi merupakan metrik utama yang digunakan untuk mengukur performa model klasifikasi yang dibangun. Nilai akurasi menunjukkan proporsi prediksi yang sesuai dengan label sebenarnya dari total data uji.

$$
\text{Accuracy} = \frac{\text{Jumlah Prediksi Benar}}{\text{Total Jumlah Data}} = \frac{TP + TN}{TP + TN + FP + FN}
$$

**Penjelasan**

- **TP**: True Positives (prediksi benar untuk kelas positif)
- **TN**: True Negatives (prediksi benar untuk kelas negatif)
- **FP**: False Positives (prediksi salah untuk kelas positif)
- **FN**: False Negatives (prediksi salah untuk kelas negatif)

Untuk *multi-class classification*, kita tidak lagi pakai TP/TN/FP/FN secara eksplisit, tapi rumus sederhananya tetap:

$$
\text{Accuracy} = \frac{\text{Jumlah label yang diprediksi dengan benar}}{\text{Total jumlah sampel}}
$$

|          |    KNN   | RandomForest | LogisticRegression |
|---------:|:--------:|:------------:|:------------------:|
| accuracy | 0.970455 |   0.988636   |      0.938636      |

<p>Tabel 5. Hasil akurasi pada tiap model</p>

Hasil evaluasi akurasi pada tabel 5 tentang akurasi pada masing-masing model dapat dijelaskan sebagai berikut:

- Random Forest memperoleh akurasi tertinggi yaitu 98.8%, diikuti oleh ***KNN*** dengan 97.0%, dan ***Logistic Regression*** dengan 93.9%.
- Ketiga model menghasilkan akurasi yang tinggi (>93%), menunjukkan bahwa model mampu mengenali pola dari fitur-fitur seperti **N**, **P**, **K**, **suhu**, **kelembaban**, *pH*, dan **curah hujan** dengan baik.
- Dengan akurasi tersebut, model telah memenuhi tujuan utama proyek, yaitu membangun sistem prediktif yang mampu merekomendasikan jenis tanaman secara akurat.

### Confusion Matrix

Untuk memahami lebih detail performa model, digunakan **Confusion Matrix** sebagai visualisasi kesalahan klasifikasi antar kelas.

<p align="center">
  <img src="https://raw.githubusercontent.com/J3ndra/dicoding_machine_learning_terapan/refs/heads/main/Predictive%20Analytics/Images/image_7.png" />
</p>

<p align="center">Gambar 7. Hasil confusion matrix</p>

Gambar 7 memperlihatkan konsistensi tinggi pada prediksi model ***Random Forest*** dan ***KNN***, dengan tingkat kesalahan klasifikasi yang rendah di hampir semua kelas, menunjukkan keandalannya dalam membedakan kebutuhan tanaman berdasarkan kondisi tanah dan iklim; **ini secara langsung menjawab problem statement tentang kesulitan pemilihan tanaman optimal berbasis data dengan meminimalkan risiko salah tanam melalui rekomendasi akurat.**

## Kesimpulan

Dapat disimpulkan model yang dibangun memberikan rekomendasi berbasis data yang mampu menggantikan pendekatan tradisional berbasis intuisi. Dengan akurasi tinggi, model ini mampu meningkatkan efisiensi pertanian melalui rekomendasi tanaman yang optimal, sekaligus mengurangi risiko inefisiensi sumber daya. Model ini memberikan nilai tambah sebagai alat bantu keputusan berbasis *Machine Learning*, yang berpotensi diintegrasikan ke dalam sistem pertanian presisi (*precision agriculture*) [3].

## Referensi

[1] Naidu, N. B. R. (2024). Harvest Harmony: Integrating linear and nonlinear machine learning models for precision crop recommendation. Deleted Journal, 32(1s), 585–610. https://doi.org/10.52783/cana.v32.2346

[2] Dahiphale, D., Shinde, P., Patil, K., & Dahiphale, V. (2023). Smart Farming: Crop recommendation using Machine learning with challenges and future ideas. JOURNAL OF IEEE TRANSACTIONS ON ARTIFICIAL INTELLIGENCE. https://doi.org/10.36227/techrxiv.23504496

[3] Kakade, S., Kulkarni, R., Dhawale, S., & C, M. F. (2023). Utilization of Machine Learning Algorithms for Precision Agriculture: Enhancing Crop Selection. Green Intelligent Systems and Applications, 3(2), 86–97. https://doi.org/10.53623/gisa.v3i2.313