# Laporan Proyek Machine Learning - Daniel Alvaro Sormin

## Project Overview

Pembuatan proyek rekomendasi _film_ menjadi semakin penting seiring dengan perkembangan pesat industri hiburan dan popularitas _platform_ _streaming_. Saat ini, kita hidup di era di mana akses ke konten digital, termasuk _film_ dan acara TV, semakin mudah dan melimpah. Sementara itu, jumlah judul yang tersedia di berbagai platform _streaming_ seperti Netflix, Amazon Prime, dan Hulu terus meningkat. Dalam situasi di mana banyaknya pilihan bisa menjadi suatu tantangan, proyek rekomendasi _film_ muncul sebagai solusi yang efektif untuk membantu pengguna menemukan konten yang sesuai dengan preferensi dan minat mereka.

Rekomendasi _film_ bukan hanya tentang menyajikan daftar _film_ yang populer, tetapi juga tentang memahami preferensi individual pengguna. Pendekatan ini memanfaatkan kecerdasan buatan (AI) dan teknik-teknik machine learning untuk menganalisis pola perilaku pengguna, sejarah penontonannya, serta karakteristik _film_ itu sendiri. Melalui penggunaan algoritma-algoritma yang canggih, seperti _collaborative filtering_ atau _content-based filtering_, proyek rekomendasi _film_ dapat memberikan saran yang personal dan relevan.

Dari perspektif bisnis, memiliki sistem rekomendasi yang kuat dapat meningkatkan retensi pengguna, memperpanjang waktu penonton, dan memperkuat loyalitas terhadap platform. Ini juga memberikan manfaat bagi industri _film_ secara keseluruhan dengan membantu _film_-_film_ yang mungkin kurang dikenal mendapatkan eksposur lebih besar.

Dengan pertumbuhan pesat data dan kecerdasan buatan, proyek rekomendasi _film_ menjadi area yang menarik untuk dieksplorasi dan dikembangkan lebih lanjut. Melalui pendekatan ini, harapannya adalah bahwa pengguna dapat menemukan konten yang mereka nikmati tanpa harus menggali terlalu dalam ke dalam katalog yang semakin luas. Sebagai kontribusi pada tren transformasi digital dalam dunia hiburan, proyek ini tidak hanya menghadirkan solusi praktis tetapi juga memperkaya pengalaman menonton bagi penonton modern yang selalu mencari konten yang relevan dan menarik.

## Business Understanding

Proyek ini dibangun agar memberikan gambaran kepada individu/perusahaan yang ingin mengembangkan sistem rekomendasi _film_ menggunakan teknik _Content-Based Filtering_ dan _Collaborative Filtering_.

### Problem Statements

- Tantangan Keragaman Preferensi: Bagaimana mengatasi kesulitan dalam memahami dan memenuhi preferensi yang sangat beragam dari pengguna dalam sistem rekomendasi _film_? Contoh Kasus: Sebuah _platform streaming film_ memiliki pengguna dari berbagai kelompok usia, latar belakang budaya, dan preferensi _film_ yang sangat beragam. Seorang pengguna dapat menjadi penggemar _film_ aksi Hollywood, sementara yang lain mungkin lebih tertarik pada _film_ animasi Jepang atau drama Eropa. Tantangannya adalah bagaimana sistem rekomendasi dapat memahami dan memenuhi preferensi yang sangat beragam ini, sehingga setiap pengguna merasa sistem memberikan rekomendasi yang relevan dengan selera mereka.
- Perubahan Dinamis dalam Tren Konten: Bagaimana membangun sistem rekomendasi yang dapat mengikuti perubahan dinamis dalam tren konten _film_ dan tetap memberikan rekomendasi yang relevan? Contoh Kasus: Industri _film_ terus berubah dengan cepat, dan tren konten bisa berubah dari waktu ke waktu. Misalnya, munculnya genre baru atau popularitas aktor tertentu yang sedang naik daun dapat mempengaruhi tren konten. Sistem rekomendasi harus dapat mendeteksi perubahan-perubahan ini dan secara dinamis memperbarui modelnya sehingga tetap memberikan rekomendasi yang relevan dengan tren konten terkini. Sebagai contoh, jika _film_-_film_ superhero sedang naik daun, sistem harus dapat mengidentifikasi perubahan ini dan meningkatkan bobotnya pada jenis _film_ tersebut.
- Analisis Konten yang Mendalam: Bagaimana meningkatkan pemahaman sistem terhadap preferensi pengguna melalui analisis konten _film_ secara mendalam? Contoh Kasus: Sebuah platform streaming ingin memberikan pengalaman yang lebih personal kepada pengguna dengan memahami lebih dalam elemen-elemen yang disukai atau tidak disukai dalam suatu _film_. Sebagai contoh, jika seorang pengguna menyukai _film_ dengan pengembangan karakter yang kuat, twist plot yang tak terduga, dan sinematografi yang indah, sistem perlu menganalisis konten _film_ secara mendalam. Ini bisa mencakup analisis dialog, pemahaman struktur naratif, dan evaluasi visual untuk menyampaikan rekomendasi yang lebih akurat berdasarkan preferensi spesifik pengguna tersebut.

### Goals

- Mengembangkan sistem rekomendasi yang mampu memahami dan merespons secara akurat terhadap preferensi _film_ yang sangat beragam dari pengguna.
- Membangun model rekomendasi yang dapat secara dinamis menyesuaikan diri dengan perubahan tren konten _film_ dan tetap relevan.
- Meningkatkan analisis konten _film_ agar sistem dapat lebih memahami elemen-elemen yang disukai oleh pengguna.

### Solution statements

- Menerapkan sistem _content-based filtering_ yang menggunakan analisis konten _film_, seperti genre, aktor, sutradara, dan plot, untuk memberikan rekomendasi. Ini memungkinkan sistem untuk memahami preferensi pengguna berdasarkan kesamaan dengan konten yang disukai sebelumnya.
- Menerapkan metode _collaborative filtering_ yang memanfaatkan informasi dari pengguna lain untuk memberikan rekomendasi. Sistem akan mempertimbangkan perilaku dan preferensi pengguna serupa untuk memprediksi _film_ yang mungkin disukai.

## Data Understanding

Dataset yang digunakan pada project _machine learning_ untuk membuat sistem rekomendasi _film_ berasal dari [The Movies Dataset - Kaggle](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset). Adapun dataset yang digunakan yaitu credits.csv, movies.csv dan ratings_small.csv.

Berikut merupakan informasi detail pada dataset:

1. Dataset credits
   - Dataset memiliki format csv
   - Dataset memiliki 4803 sampel dengan 4 fitur
   - Dataset memiliki 1 fitur bertipe integer dan 3 fitur bertipe object
2. Dataset movies
   - Dataset memiliki format csv
   - Dataset memiliki 4803 sampel dengan 20 fitur
   - Dataset memiliki 4 fitur bertipe integer, 3 fitur bertipe float dan 17 fitur bertipe object
3. Dataset ratings_small
   - Dataset memiliki format csv
   - Dataset memiliki 100004 sampel dengan 4 fitur
   - Dataset memiliki 3 fitur bertipe integer dan 1 fitur bertipe float

Variabel-variabel pada "The Movies Dataset" adalah sebagai berikut:

1. Dataset _credit_

   - movie_id: Identifier unik untuk setiap _film_ dalam dataset.
   - title: Judul _film_.
   - cast: Informasi tentang pemeran dalam _film_, mungkin berupa daftar nama pemeran.
   - crew: Informasi tentang kru produksi _film_, seperti sutradara, penulis, dan anggota staf lainnya.

2. Dataset _movies_

   - budget: Anggaran produksi _film_.
   - genres: Genre atau kategori _film_, bisa berupa beberapa genre.
   - homepage: URL halaman utama atau situs web _film_.
   - id: Identifier unik untuk setiap _film_ dalam dataset.
   - keywords: Kata kunci atau istilah terkait _film_.
   - original_language: Bahasa asli _film_.
   - original_title: Judul asli _film_ (dalam bahasa asli).
   - overview: Ringkasan atau deskripsi singkat tentang _film_.
   - popularity: Tingkat popularitas _film_, mungkin berdasarkan sejumlah faktor.
   - production_companies: Perusahaan-perusahaan yang terlibat dalam produksi _film_.
   - production_countries: Negara-negara di mana _film_ diproduksi atau disyut.
   - release_date: Tanggal rilis _film_.
   - revenue: Pendapatan kotor atau hasil keuangan dari _film_.
   - runtime: Durasi atau lama _film_.
   - spoken_languages: Bahasa-bahasa yang digunakan dalam _film_.
   - status: Status produksi _film_ (misalnya, "Released").
   - tagline: Frasa atau kalimat pemasaran yang mencirikan _film_.
   - title: Judul _film_.
   - vote_average: Rata-rata peringkat atau nilai yang diberikan oleh penonton.
   - vote_count: Jumlah total peringkat yang diberikan oleh penonton.

3. Dataset _ratings_small_
   - userId: _Identifier_ unik untuk setiap pengguna.
   - movieId: _Identifier_ unik untuk setiap _film_.
   - rating: Peringkat yang diberikan oleh pengguna untuk suatu _film_.
   - timestamp: Waktu ketika peringkat diberikan, mungkin dalam format timestamp.

Teknik yang digunakan pada tahapan data _understanding_:

- Variabel _Description_: Untuk memeriksa setiap variabel yang terdapat dalam dataset apakah terdapat nilai NaN, duplicate, dll.
- _Univariate Analysis_: Untuk menganalisis setiap variabel secara terpisah dengan fokus pada distribusi nilai, statistik deskriptif, dan visualisasi.

Tabel 1. Analisis fitur _categorical_
  
|       | budget       | id        | popularity | revenue       | runtime | vote_average | vote_count |
|-------|--------------|-----------|------------|---------------|---------|--------------|------------|
| count |      4803.00 |   4803.00 |    4803.00 |       4803.00 | 4801.00 |      4803.00 |    4803.00 |
|  mean |  29045039.88 |  57165.48 |      21.49 |   82260638.65 |  106.88 |         6.09 |     690.22 |
|  std  |  40722391.26 |  88694.61 |      31.82 |  162857100.94 |   22.61 |         1.19 |    1234.59 |
|  min  |         0.00 |      5.00 |       0.00 |          0.00 |    0.00 |         0.00 |       0.00 |
|  25%  |    790000.00 |   9014.50 |       4.67 |          0.00 |   94.00 |         5.60 |      54.00 |
|  50%  |  15000000.00 |  14629.00 |      12.92 |   19170001.00 |  103.00 |         6.20 |     235.00 |
|  75%  |  40000000.00 |  58610.50 |      28.31 |   92917187.00 |  118.00 |         6.80 |     737.00 |
|  max  | 380000000.00 | 459488.00 |     875.58 | 2787965087.00 |  338.00 |        10.00 |   13752.00 |


  Berikut merupakan visualisasi analisis _numerical_:
  
  <img width="600" src="https://github.com/danalvr/MLT-Assignment-Dicoding/assets/81479217/5d7aceee-164b-40bc-a41f-c9fe64451853" alt="gambar analisis numerik" />

  Gambar 1. Visualiasi analisis numerical

Visual tersebut menampilkan data fitur _numerical_ untuk mempermudah dalam meilhat persebaran data yang ada. Dalam visual data tersebut kita berfokus pada fitur _vote average_, _popularity_ dan _vote count_ yang merupakan fitur yang relevan dengan sistem rekomendasi yang dibuat. Pada fitur _vote average_ populasi data terbanyak terdapat pada rentang nilai 6-8. Pada fitur _popularity_ persebaran data terbanyak dengan _value_ diatas 2500. Terakhir pada fitur _vote count_ jumlah data terbanyak lebih dari 2500. Analisis lebih mendalam menunjukkan bahwa rentang nilai 6-8 pada fitur _vote average_ mengindikasikan bahwa sebagian besar _film_ memiliki penilaian yang positif dari pengguna, tetapi belum tentu mendapatkan popularitas yang tinggi. Terkait persebaran data _popularity_ di atas 2500, ini dapat diartikan bahwa sebagian besar _film_ dalam dataset memiliki tingkat popularitas yang relatif tinggi, mungkin karena telah mendapatkan perhatian luas dari penonton. Sementara itu, jumlah data terbanyak pada fitur _vote count_ yang lebih dari 2500 menunjukkan bahwa sejumlah besar _film_ dalam dataset memiliki jumlah pemilih atau pengulas yang signifikan.

Berdasarkan analisis yang dilakukan terhadap fitur kategori dan numerik dapat disimpulkan data memiliki kualitas bagus karena tidak terdapat _outlier_. Namun, terdapat _missing value_ pada fitur _homepage_, _overview_, _release_date_, _runtime_, dan _tagline_ yang akan dijelaskan lebih rinci pada tahap _data preparation_.

## Data Preparation

Dalam pembuatan sistem rekomendasi _film_ terdapat beberapa bagian dalam tahapan _data preparation_ yaitu _assessing variabel_ dan _demographic filtering_. _Assessing variabel_ digunakan untuk menjaga kualitas dataset yang akan digunakan dalam pembuatan sistem rekomendasi sedangkan _demographic filtering_ digunakan untuk memastikan fitur tersebut benar-benar penting dalam pembuatan sistem rekomendasi _film_. Berikut merupakan penjelasan secara rinci tahapan dari _assessing variabel_ dan _demographic filtering_.

### Assessing Variabel
Pada tahap _data preparation_ langkah awal yang dilakukan adalah pengecekan dataset yang telah digabung sebelumnya. Setelah dilakukan pengecekan terhadap dataset terdapat beberapa _missing value_ pada fitur _homepage_, _overview_, _release_date_, _runtime_, dan _tagline_. Namun, _missing value_ tersebut memiliki entri data yang banyak sehingga data tidak akan dihapus karena memiliki informasi yang penting. Setelah dilakukan pengecekan lebih lanjut solusi yang tepat adalah dengan mengganti nilai _missing value_ dengan nilai string kosong karena fitur tersebut tidak terlalu berpengaruh terhadap sistem rekomendasi.

### Demographic Filtering:

Pada bagian _demographic filtering_ akan dilakukan pemetaan terhadap dataset dengan membuat sebuah fitur baru bernama _score_. Fitur _score_ tersebut akan digunakan sebagai parameter sistem rekomendasi yang akan dikalkulasikan untuk melihat kesamaan data _film_ pada teknik _content-based filtering_. Berikut merupakan rumus yang digunakan untuk menghitung WR atau fitur _score_: 

<img width="400" src="https://github.com/danalvr/MLT-Assignment-Dicoding/assets/81479217/c2e5a6df-9a2e-4e29-ade5-65e9c810298a" alt="gambar rumus WR" />

Gambar 2. Rumus menghitung fitur _score_

Keterangan:

- v adalah jumlah vote untuk _film_ tersebut
- m adalah jumlah vote minimum yang diperlukan untuk dicantumkan dalam bagan
- R adalah total rata-rata vote _film_
- C adalah rata-rata vote di seluruh laporan

## Modeling

Membuat sistem rekomendasi dengan mengimplementasikan teknik _content-based filtering_ dan _collaborative filtering_. Adapun kelebihan dari teknik yang digunakan adalah sebagai berikut:

### Content-based Filtering

Kelebihan:

- Personalisasi yang Baik: _Content-based filtering_ dapat memberikan rekomendasi yang sangat personal karena didasarkan pada preferensi dan karakteristik pengguna yang sudah diketahui.
- Sensitivitas terhadap Perubahan: Model cenderung stabil terhadap perubahan dalam perilaku pengguna karena bergantung pada konten atau fitur item yang jarang berubah.
- Interpretabilitas Tinggi: Model ini cenderung lebih mudah diinterpretasikan karena rekomendasi didasarkan pada fitur-fitur yang dapat dimengerti dan dijelaskan.

Kekurangan:

- Keterbatasan dalam Diversitas: Tidak mampu memberikan rekomendasi yang sangat berbeda dari preferensi pengguna saat ini karena terlalu fokus pada karakteristik item yang telah disukai.
- Keterbatasan dalam Menangani Data Baru: Kesulitan dalam menangani item baru yang belum memiliki sejarah interaksi karena tidak ada data kollaboratif yang dapat digunakan.
- Ketergantungan pada Representasi Fitur: Kualitas rekomendasi sangat bergantung pada kualitas representasi fitur item dan profil pengguna.

Implementasi:
_Content-based filtering_ digunakan untuk memberikan sebuah rekomendasi _film_ yang memiliki kesamaan fitur dengan parameter berupa _overview_, _cast_, _crew_, _keyword_, _tagline_, dsb. Misal, ketika _user_ sedang menonton atau sedang mencari sebuah _film_ maka sistem akan memberikan rekomendasi _film_ yang mirip dengan _film_ yang telah ditonton oleh _user_ ataupun yang sedang dicari oleh _user_ baik dari segi _genre_, _overview_ ataupun sutradara dari _film_ tersebut. Untuk implementasi teknik _content-based filtering_ langkah awal yang dilakukan adalah merepresentasikan fitur _overview_ menjadi sebuah matriks menggunakan metode TF-IDF dengan fungsi TfidfVectorizer dari _library_ sklearn__. Seperti yang diketahui TF-IDF (Term Frequency-Inverse Document Frequency) merupakan metode pemodelan yang digunakan untuk mengekstraksi fitur dari teks. Hasil dari _tfidf_matrix_ akan digunakan nantinya sebagai fitur masukan dalam sistem rekomendasi. Langkah selanjutnya yaitu menghitung derajat kesamaan antar _film_ dengan menggunakan fungsi _cosine_similarity_ dari library sklearn. Selanjutnya akan dibuat sebuah fungsi _get_recommendation_ dengan parameter berupa _title_ dan nilai dari _cosine similarity_ untuk menampilkan rekomendasi _film_ yang memiliki kemiripan dengan _film_ yang telah ditonton atau yang sedang dicari.

Berikut merupakan hasil rekomendasi _film_ menggunakan teknik _content-based filtering_ yang memiliki kemiripan dengan _film_ "Batman":

Tabel 2. Hasil rekomendasi _film_ dengan _content-based filtering_

| id   | title                                   |
|------|-----------------------------------------|
| 65   | The Dark Knight                         |
|  299 |                          Batman Forever |
|  428 |                          Batman Returns |
| 1359 |                                  Batman |
| 3854 | Batman: The Dark Knight Returns, Part 2 |
|  119 |                           Batman Begins |
| 2507 |                               Slow Burn |
|   9  |      Batman v Superman: Dawn of Justice |
| 1181 |                                     JFK |
| 210  | Batman & Robin                          |

Hasil rekomendasi _film_ tersebut diberikan oleh sistem berdasarkan kesamaan oleh paramater _overview_, genre, _keyword_, _director_ _cast_ dan _crew_.

### Collaborative Filtering

Kelebihan:

- Dapat Menangani Data Baru: Mampu memberikan rekomendasi untuk item baru atau item yang jarang terlihat karena mengandalkan pola kollaboratif antar pengguna.
- Diversitas yang Tinggi: Cenderung memberikan rekomendasi yang lebih beragam karena bergantung pada pola interaksi pengguna secara keseluruhan.
- Tidak Memerlukan Informasi Fitur: Tidak memerlukan informasi eksternal tentang item atau pengguna karena hanya memerlukan data kollaboratif.

Kekurangan:

- _Cold Start Problem_: Rentan terhadap Cold Start Problem karena membutuhkan data interaksi yang cukup sebelum dapat memberikan rekomendasi yang akurat.
- _Sparsity_: Jika dataset sangat besar, bisa terjadi sparsity, di mana sebagian besar pengguna hanya berinteraksi dengan sebagian kecil item, membuat model sulit memberikan rekomendasi yang baik.
- _Scalability_: Kesulitan dalam skala besar karena memerlukan perhitungan yang kompleks untuk matriks kollaboratif, yang dapat menjadi tantangan pada dataset yang sangat besar.

Implementasi:
_Collaborative filtering_ digunakan untuk memberikan sebuah rekomendasi _film_ berdasarkan informasi yang diberikan oleh _user_ lain. Dalam hal ini digunakan suatu pendekatan yaitu _user-based filtering_ untuk memberikan rekomendasi berupa _film_ berdasarkan kesamaan preferensi dengan _user_ lain.  Misal, _user_ A dan _user_ B menyukai beberapa _film_ dengan _genre_ yang sama. Jika _user_ A menyukai _film_ A maka sistem akan memberikan rekomendasi berupa _film_ A kepada _user_ B. Untuk implementasi teknik _collaborative filtering_ langkah awal yang dilakukan adalah melakukan _encoding_ pada fitur _userId_ dan _movieId_. Kemudian, hasil _encoding_ akan dipetakan pada variabel _ratings_df['user']_ dan _ratings_df['movie']_. Kemudian, dilakukan _split_ dataset menjadi data latih dan data uji dengan rasio 8:2. Kemudian, akan dibuat sebuah kelas _RecommenderNet_ dengan parameter _tf.keras.Model_ yang merupakan variabel bawaan yang diimport dari _framework_ TensorFlow. _Recommendernet_ merupakan implementasi dari model _deep learning_ yang memanfaatkan pembelajaran _embedding_ untuk menyandikan pengguna dan _item_ dalam representasi yang lebih kompak, dan kemudian memanfaatkan _dot product_ dan bias untuk menghasilkan prediksi. Kemudian, akan dibuat sebuah variabel model dengan menginisiasi kelas _RecommenderNet_. Kemudian model tersebut akan di-_compile_. Kemudian akan ditraining dengan iterasi (_epochs_) sebanyak 10 kali.

Berikut merupakan hasil rekomendasi _film_ dengan _userId_ yaitu 461:

Tabel 3. Hasil rekomendasi _film_ dengan _collaborative filtering_ pada _userId_ 461

| title                                            | ratings |
|--------------------------------------------------|---------|
| American History X                               |     8.2 |
|                Batman Begins                     |     7.5 |
|         Terminator 2: Judgment Day               |     7.7 |
|           Raiders of the Lost Ark                |     7.7 |
|               Apocalypse Now                     |     8.0 |
| Pirates of the Caribbean: Dead Man's Chest       |     7.0 |
|             Mission: Impossible                  |     6.7 |
|            Ice Age: The Meltdown                 |     6.5 |
|               Horrible Bosses                    |     6.5 |
| Terminator Salvation                             |     5.9 |

Hasil rekomendasi tersebut diberikan oleh sistem dikarenakan _userId_ 461 memberikan _rating_ yang tinggi terhadap beberapa _genre_ _film_ dimana _user_ lain juga memberikan _rating_ yang sama sehingga sistem memberikan rekomendasi _film_ berdasarkan _rating_ tinggi yang diberikan oleh user lain.

## Evaluation

Pada tahap _evaluation_ akan dilakukan analisis terhadap sistem rekomendasi yang dibuat dengan menggunakan teknik _content-based filtering_ dan _collaborative filtering_.

### _Content-Based Filtering_
Evaluasi akan dilakukan dengan mencoba memasukkan _inputan_ sebuah _film_ berjudul "Avatar". Setelah itu sistem menampilkan list beberapa rekomendasi _film_ yang relevan. Kemudian akan dihitung jumlah rekomendasi _film_ yang relevan menggunakan rumus _precision_. Rumus _precision_ akan mengukur seberapa akurat sistem dalam memberikan rekomendasi _film_ dengan menyatakan rasio _item_ relevan yang dihasilkan oleh sistem terhadap total _item_ yang dihasilkan. Berdasarkan percobaan dengan menginput judul _film_ "Avatar" sistem menampilkan 10 item _film_ yang relevan dengan menganalisis secara manual dengan cara mengecek _detail_ masing-masing dari _film_ tersebut. Hasil analisis manual tersebut yaitu sistem berhasil menampilkan rekomendasi _film_ yang relevan. Hal ini dibuktikan dari hasil _film_ 'Avatar' yang memiliki genre yang sama dengan yang direkomendasikan oleh sistem.

Berikut merupakan rumus dari _precision_:

<img width="400" src="https://github.com/danalvr/MLT-Assignment-Dicoding/assets/81479217/38bf2d9a-dec7-4121-95cd-983e628c10e1" alt="Hasil rekomendasi _film_" />

Gambar 3. Rumus _precision_

Berikut merupakan hasil rekomendasi _film_ yang relevan dengan _film_ "Avatar":

Tabel 4. Hasil rekomendasi _film_ relevan dengan film "Avatar"  menggunakan _collaborative filtering_

| id   | title                        | genres                             |
|------|------------------------------|------------------------------------|
| 3604 |                    Apollo 18 | [horror, thriller, sciencefiction] |
| 2130 |                 The American |           [crime, drama, thriller] |
|  634 |                   The Matrix |           [action, sciencefiction] |
| 1341 |         The Inhabited Island |  [action, fantasy, sciencefiction] |
|  529 |             Tears of the Sun |               [action, drama, war] |
| 1610 |                        Hanna |      [action, thriller, adventure] |
|  311 | The Adventures of Pluto Nash |   [action, comedy, sciencefiction] |
|  847 |                     Semi-Pro |                           [comedy] |
|  775 |                    Supernova | [horror, sciencefiction, thriller] |
| 2628 |          Blood and Chocolate |           [drama, fantasy, horror] |

Seperti yang diketahui bahwa _film_ "Avatar" memiliki genre [action, adventure, fantasy]. Berdasarkan tabel berikut terdapat 6 _film_ yang direkomendasikan memiliki kesamaan genre dengan _film_ "Avatar" yaitu, _film_ "The Matrix", "The Inhabitated Island", "Tears of The Sun", "Hanna", "The Adventures of Pluto Nash" dan "Blood and Chocolate". Artinya tingkat akurasi sistem yang ditinjau dari fitur genre mencapai 60% yang mana hasil ini menunjukkan bahwa sistem telah bekerja dengan baik dalam memberikan rekomendasi _film_. 

### _Collaborative filtering_

Evaluasi akan dilakukan dengan menggunakan metrik RMSE pada saat melakukan _training_ model. Cara kerja metrik RMSE yaitu dengan menghitung selisih antara nilai aktual dan nilai prediksi. Kemudian setiap selisih tersebut akan dikuadratkan dan hitung rata-rata-nya yang kemudian dilakukan akar kuadrat dari hasil rata-rata tersebut. Semakin kecil nilai RMSE, semakin baik kinerja dari model sistem rekomendasi yang dibuat.

Berikut merupakan rumus RMSE:

<img width="400" src="https://github.com/danalvr/MLT-Assignment-Dicoding/assets/81479217/454904ca-65f8-4c92-b1ba-eea865ca2b56" alt="Metrik RMSE" />

Gambar 4. Rumus RMSE

Berikut merupakan nilai hasil Metrik RMSE pada data latih dan data uji:

Tabel 5. Hasil data latih dan data uji model _collaborative filtering_ menggunakan metrik RMSE

| epochs | data latih | data uji |
|--------|------------|----------|
|    1   |     0.2281 |   0.2112 |
|    2   |     0.2037 |   0.2063 |
|    3   |     0.1983 |   0.2049 |
|    4   |     0.1952 |   0.2039 |
|    5   |     0.1927 |   0.2033 |
|    6   |     0.1913 |   0.2033 |
|    7   |     0.1901 |   0.2032 |
|    8   |     0.1890 |   0.2030 |
|    9   |     0.1883 |   0.2023 |
|   10   |     0.1871 |   0.2032 |

Berikut merupakan hasil visualisasi data latih dan data uji menggunakan metrik RMSE:

<img width="400" src="https://github.com/danalvr/MLT-Assignment-Dicoding/assets/81479217/1a11b5f1-5a62-47be-be21-ca45c3fe4d28" alt="Metrik RMSE" />

Gambar 5. Visualisasi data latih dan data uji dengan metrik RMSE

Dalam analisis grafik diatas, dapat diketahui bahwa model memiliki kinerja yang relatif tinggi pada data latih. Hal ini mungkin disebabkan oleh sampel data yang sangat mirip atau cocok dengan sampel data uji. Namun, dari hasil uji coba, kinerja model terhadap data baru ternyata relatif rendah, sehingga memerlukan peningkatan kinerja pada data baru. Hal ini mengindikasikan bahwa model mengalami _overfitting_. Untuk mengoptimalkan model agar dapat memberikan rekomendasi yang lebih baik dapat dilakukan penyesuaian _hyperparameter_ untuk meningkatkan performa model.

Kesimpulan dari evaluasi yang dilakukan adalah sistem yang dibangun menggunakan teknik _content-based filtering_ dan _collaborative filtering_ dapat memberikan rekomendasi _film_ yang relevan dan dapat merespons secara akurat terhadap preferensi _film_ yang sangat beragam dari pengguna. Namun, pada sistem rekomendasi yang dikembangkan menggunakan teknik _collaborative filtering_ terdapat indikasi _overfitting_ sehingga perlu dilakukan optimalisasi pada model tersebut.
