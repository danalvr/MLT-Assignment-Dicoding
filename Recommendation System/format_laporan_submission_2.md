![image](https://github.com/danalvr/MLT-Assignment-Dicoding/assets/81479217/a26609c9-731f-49b6-b5a4-7695d417ec5a)# Laporan Proyek Machine Learning - Daniel Alvaro Sormin

## Project Overview

Pembuatan proyek rekomendasi film menjadi semakin penting seiring dengan perkembangan pesat industri hiburan dan popularitas platform streaming. Saat ini, kita hidup di era di mana akses ke konten digital, termasuk film dan acara TV, semakin mudah dan melimpah. Sementara itu, jumlah judul yang tersedia di berbagai platform streaming seperti Netflix, Amazon Prime, dan Hulu terus meningkat. Dalam situasi di mana banyaknya pilihan bisa menjadi suatu tantangan, proyek rekomendasi film muncul sebagai solusi yang efektif untuk membantu pengguna menemukan konten yang sesuai dengan preferensi dan minat mereka.

Rekomendasi film bukan hanya tentang menyajikan daftar film yang populer, tetapi juga tentang memahami preferensi individual pengguna. Pendekatan ini memanfaatkan kecerdasan buatan (AI) dan teknik-teknik machine learning untuk menganalisis pola perilaku pengguna, sejarah penontonannya, serta karakteristik film itu sendiri. Melalui penggunaan algoritma-algoritma yang canggih, seperti collaborative filtering atau content-based filtering, proyek rekomendasi film dapat memberikan saran yang personal dan relevan.

Dari perspektif bisnis, memiliki sistem rekomendasi yang kuat dapat meningkatkan retensi pengguna, memperpanjang waktu penonton, dan memperkuat loyalitas terhadap platform. Ini juga memberikan manfaat bagi industri film secara keseluruhan dengan membantu film-film yang mungkin kurang dikenal mendapatkan eksposur lebih besar.

Dengan pertumbuhan pesat data dan kecerdasan buatan, proyek rekomendasi film menjadi area yang menarik untuk dieksplorasi dan dikembangkan lebih lanjut. Melalui pendekatan ini, harapannya adalah bahwa pengguna dapat menemukan konten yang mereka nikmati tanpa harus menggali terlalu dalam ke dalam katalog yang semakin luas. Sebagai kontribusi pada tren transformasi digital dalam dunia hiburan, proyek ini tidak hanya menghadirkan solusi praktis tetapi juga memperkaya pengalaman menonton bagi penonton modern yang selalu mencari konten yang relevan dan menarik.

## Business Understanding

Proyek ini dibangun agar memberikan gambaran kepada individu/perusahaan yang ingin mengembangkan sistem rekomendasi film menggunakan teknik Content-Based Filtering dan Collaborative Filtering.

### Problem Statements

- Tantangan Keragaman Preferensi: Bagaimana mengatasi kesulitan dalam memahami dan memenuhi preferensi yang sangat beragam dari pengguna dalam sistem rekomendasi film? Contoh Kasus: Sebuah platform streaming film memiliki pengguna dari berbagai kelompok usia, latar belakang budaya, dan preferensi film yang sangat beragam. Seorang pengguna dapat menjadi penggemar film aksi Hollywood, sementara yang lain mungkin lebih tertarik pada film animasi Jepang atau drama Eropa. Tantangannya adalah bagaimana sistem rekomendasi dapat memahami dan memenuhi preferensi yang sangat beragam ini, sehingga setiap pengguna merasa sistem memberikan rekomendasi yang relevan dengan selera mereka.
- Perubahan Dinamis dalam Tren Konten: Bagaimana membangun sistem rekomendasi yang dapat mengikuti perubahan dinamis dalam tren konten film dan tetap memberikan rekomendasi yang relevan? Contoh Kasus: Industri film terus berubah dengan cepat, dan tren konten bisa berubah dari waktu ke waktu. Misalnya, munculnya genre baru atau popularitas aktor tertentu yang sedang naik daun dapat mempengaruhi tren konten. Sistem rekomendasi harus dapat mendeteksi perubahan-perubahan ini dan secara dinamis memperbarui modelnya sehingga tetap memberikan rekomendasi yang relevan dengan tren konten terkini. Sebagai contoh, jika film-film superhero sedang naik daun, sistem harus dapat mengidentifikasi perubahan ini dan meningkatkan bobotnya pada jenis film tersebut.
- Analisis Konten yang Mendalam: Bagaimana meningkatkan pemahaman sistem terhadap preferensi pengguna melalui analisis konten film secara mendalam? Contoh Kasus: Sebuah platform streaming ingin memberikan pengalaman yang lebih personal kepada pengguna dengan memahami lebih dalam elemen-elemen yang disukai atau tidak disukai dalam suatu film. Sebagai contoh, jika seorang pengguna menyukai film dengan pengembangan karakter yang kuat, twist plot yang tak terduga, dan sinematografi yang indah, sistem perlu menganalisis konten film secara mendalam. Ini bisa mencakup analisis dialog, pemahaman struktur naratif, dan evaluasi visual untuk menyampaikan rekomendasi yang lebih akurat berdasarkan preferensi spesifik pengguna tersebut.

### Goals

- Mengembangkan sistem rekomendasi yang mampu memahami dan merespons secara akurat terhadap preferensi film yang sangat beragam dari pengguna.
- Membangun model rekomendasi yang dapat secara dinamis menyesuaikan diri dengan perubahan tren konten film dan tetap relevan.
- Meningkatkan analisis konten film agar sistem dapat lebih memahami elemen-elemen yang disukai oleh pengguna.

### Solution statements

- Menerapkan sistem content-based filtering yang menggunakan analisis konten film, seperti genre, aktor, sutradara, dan plot, untuk memberikan rekomendasi. Ini memungkinkan sistem untuk memahami preferensi pengguna berdasarkan kesamaan dengan konten yang disukai sebelumnya.
- Menerapkan metode collaborative filtering yang memanfaatkan informasi dari pengguna lain untuk memberikan rekomendasi. Sistem akan mempertimbangkan perilaku dan preferensi pengguna serupa untuk memprediksi film yang mungkin disukai.

## Data Understanding & Preprocessing

Dataset yang digunakan pada project machine learning untuk membuat sistem rekomendasi film berasal dari [The Movies Dataset - Kaggle](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset). Adapun dataset yang digunakan yaitu credits.csv, movies.csv dan ratings_small.csv.

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

Variabel-variabel pada The Movies Dataset adalah sebagai berikut:

1. Dataset credit

   - movie_id: Identifier unik untuk setiap film dalam dataset.
   - title: Judul film.
   - cast: Informasi tentang pemeran dalam film, mungkin berupa daftar nama pemeran.
   - crew: Informasi tentang kru produksi film, seperti sutradara, penulis, dan anggota staf lainnya.

2. Dataset movies

   - budget: Anggaran produksi film.
   - genres: Genre atau kategori film, bisa berupa beberapa genre.
   - homepage: URL halaman utama atau situs web film.
   - id: Identifier unik untuk setiap film dalam dataset.
   - keywords: Kata kunci atau istilah terkait film.
   - original_language: Bahasa asli film.
   - original_title: Judul asli film (dalam bahasa asli).
   - overview: Ringkasan atau deskripsi singkat tentang film.
   - popularity: Tingkat popularitas film, mungkin berdasarkan sejumlah faktor.
   - production_companies: Perusahaan-perusahaan yang terlibat dalam produksi film.
   - production_countries: Negara-negara di mana film diproduksi atau disyut.
   - release_date: Tanggal rilis film.
   - revenue: Pendapatan kotor atau hasil keuangan dari film.
   - runtime: Durasi atau lama film.
   - spoken_languages: Bahasa-bahasa yang digunakan dalam film.
   - status: Status produksi film (misalnya, "Released").
   - tagline: Frasa atau kalimat pemasaran yang mencirikan film.
   - title: Judul film.
   - vote_average: Rata-rata peringkat atau nilai yang diberikan oleh penonton.
   - vote_count: Jumlah total peringkat yang diberikan oleh penonton.

3. Dataset ratings_small
   - userId: Identifier unik untuk setiap pengguna.
   - movieId: Identifier unik untuk setiap film.
   - rating: Peringkat yang diberikan oleh pengguna untuk suatu film.
   - timestamp: Waktu ketika peringkat diberikan, mungkin dalam format timestamp.

Teknik yang digunakan pada tahapan data understanding:

- Variabel Description: Untuk memeriksa setiap variabel yang terdapat dalam dataset apakah terdapat nilai NaN, duplicate, dll.
- Univariate Analysis: Untuk menganalisis setiap variabel secara terpisah dengan fokus pada distribusi nilai, statistik deskriptif, dan visualisasi.

  Berikut merupakan hasil analisis categorical:
  
  <img width="400" src="https://github.com/danalvr/MLT-Assignment-Dicoding/assets/81479217/05a5d35c-7872-44cd-8b80-dccf60b7618c" alt="gambar analisis kategori" />


  Berikut merupakan hasil analisis numerical:
  
  <img width="400" src="https://github.com/danalvr/MLT-Assignment-Dicoding/assets/81479217/5d7aceee-164b-40bc-a41f-c9fe64451853" alt="gambar analisis numerik" />

Kemudian setelah melakukan analisis data menggunakan metode univariate analysis akan dilakukan tahapan data preprocessing dimana dataset credit.csv dan movie.csv akan digabung berdasarkan parameter movie_id pada dataset credit.

## Data Preparation

Pada tahap preparation dilakukan pengecekan dataset yang telah digabung sebelumnya. Setelah dilakukan pengecekan terhadap dataset terdapat beberapa missing value. Kemudian, missing value tersebut diganti dengan nilai string kosong.

### Demographic Filtering:

Pada bagian demographic filtering akan dilakukan pemetaan terhadap dataset dengan membuat sebuah fitur baru bernama score. Fitur score tersebut akan digunakan sebagai parameter sistem rekomendasi yang akan dikalkulasikan untuk melihat kesamaan data film pada teknik content-based filtering. Berikut merupakan rumus yang digunakan untuk menghitung WR atau fitur score: 

<img width="400" src="https://github.com/danalvr/MLT-Assignment-Dicoding/assets/81479217/c2e5a6df-9a2e-4e29-ade5-65e9c810298a" alt="gambar rumus WR" />

Keterangan:

- v adalah jumlah vote untuk film tersebut
- m adalah jumlah vote minimum yang diperlukan untuk dicantumkan dalam bagan
- R adalah total rata-rata vote film
- C adalah rata-rata vote di seluruh laporan

## Modeling

Membuat sistem rekomendasi dengan mengimplementasikan teknik content-based filtering dan collaborative filtering. Adapun kelebihan dari teknik yang digunakan adalah sebagai berikut:

### Content-based Filtering

Kelebihan:

- Personalisasi yang Baik: Content-based filtering dapat memberikan rekomendasi yang sangat personal karena didasarkan pada preferensi dan karakteristik pengguna yang sudah diketahui.
- Sensitivitas terhadap Perubahan: Model cenderung stabil terhadap perubahan dalam perilaku pengguna karena bergantung pada konten atau fitur item yang jarang berubah.
- Interpretabilitas Tinggi: Model ini cenderung lebih mudah diinterpretasikan karena rekomendasi didasarkan pada fitur-fitur yang dapat dimengerti dan dijelaskan.

Kekurangan:

- Keterbatasan dalam Diversitas: Tidak mampu memberikan rekomendasi yang sangat berbeda dari preferensi pengguna saat ini karena terlalu fokus pada karakteristik item yang telah disukai.
- Keterbatasan dalam Menangani Data Baru: Kesulitan dalam menangani item baru yang belum memiliki sejarah interaksi karena tidak ada data kollaboratif yang dapat digunakan.
- Ketergantungan pada Representasi Fitur: Kualitas rekomendasi sangat bergantung pada kualitas representasi fitur item dan profil pengguna.

Implementasi:
_Content-based filtering_ digunakan untuk memberikan sebuah rekomendasi _film_ yang memiliki kesamaan fitur dengan parameter berupa _overview_, _cast_, _crew_, _keyword_, _tagline_, dsb. Misal, ketika _user_ sedang menonton atau sedang mencari sebuah _film_ maka sistem akan memberikan rekomendasi _film_ yang mirip dengan _film_ yang telah ditonton oleh _user_ ataupun yang sedang dicari oleh _user_ baik dari segi _genre_, _overview_ ataupun sutradara dari _film_ tersebut. Untuk implementasi teknik _content-based filtering_ langkah awal yang dilakukan adalah merepresentasikan fitur _overview_ menjadi sebuah matriks menggunakan metode TF-IDF dengan fungsi TfidfVectorizer dari _library_ sklearn__. Langkah selanjutnya yaitu menghitung derajat kesamaan antar film dengan menggunakan fungsi _cosine_similarity_ dari library sklearn. Selanjutnya akan dibuat sebuah fungsi _get_recommendation_ dengan parameter berupa _title_ dan nilai dari _cosine similarity_ untuk menampilkan rekomendasi _film_ yang memiliki kemiripan dengan _film_ yang telah ditonton atau yang sedang dicari.

Berikut merupakan hasil rekomendasi film menggunakan teknik _content-based filtering_ yang memiliki kemiripan dengan _film_ "Batman":

<img width="400" src="https://github.com/danalvr/MLT-Assignment-Dicoding/assets/81479217/64bfd03c-34b0-4ebd-9b42-1c4eb2a957a4" alt="Hasil rekomendasi film" />

### Collaborative Filtering

Kelebihan:

- Dapat Menangani Data Baru: Mampu memberikan rekomendasi untuk item baru atau item yang jarang terlihat karena mengandalkan pola kollaboratif antar pengguna.
- Diversitas yang Tinggi: Cenderung memberikan rekomendasi yang lebih beragam karena bergantung pada pola interaksi pengguna secara keseluruhan.
- Tidak Memerlukan Informasi Fitur: Tidak memerlukan informasi eksternal tentang item atau pengguna karena hanya memerlukan data kollaboratif.

Kekurangan:

- Cold Start Problem: Rentan terhadap Cold Start Problem karena membutuhkan data interaksi yang cukup sebelum dapat memberikan rekomendasi yang akurat.
- Sparsity: Jika dataset sangat besar, bisa terjadi sparsity, di mana sebagian besar pengguna hanya berinteraksi dengan sebagian kecil item, membuat model sulit memberikan rekomendasi yang baik.
- Scalability: Kesulitan dalam skala besar karena memerlukan perhitungan yang kompleks untuk matriks kollaboratif, yang dapat menjadi tantangan pada dataset yang sangat besar.

Implementasi:
_Collaborative filtering_ digunakan untuk memberikan sebuah rekomendasi _film_ berdasarkan informasi yang diberikan oleh _user_ lain. Dalam hal ini digunakan suatu pendekatan yaitu _user-based filtering_ untuk memberikan rekomendasi berupa _film_ berdasarkan kesamaan preferensi dengan _user_ lain.  Misal, _user_ A dan _user_ B menyukai beberapa film dengan genre yang sama. Jika _user_ A menyukai _film_ A maka sistem akan memberikan rekomendasi berupa _film_ A kepada _user_ B. Untuk implementasi teknik _collaborative filtering_ langkah awal yang dilakukan adalah melakukan _encoding_ pada fitur _userId_ dan _movieId_. Kemudian, hasil _encoding_ akan dipetakan pada variabel _ratings_df['user']_ dan _ratings_df['movie']_. Kemudian, dilakukan _split_ dataset menjadi data latih dan data uji dengan rasio 8:2. Kemudian, akan dibuat sebuah kelas _RecommenderNet_ dengan parameter _tf.keras.Model_ yang merupakan variabel bawaan yang diimport dari _framework_ TensorFlow. Kemudian, akan dibuat sebuah model dengan menginisiasi kelas _RecommenderNet_. Kemudian model akan di-_compile_. Kemudian akan ditraining dengan iterasi (_epochs_) sebanyak 10 kali.

Berikut merupakan hasil rekomendasi _film_ dengan _userId_ yaitu 461:

<img width="400" src="https://github.com/danalvr/MLT-Assignment-Dicoding/assets/81479217/34abbf64-a603-4dda-9d64-7d66004f7b64" alt="Hasil rekomendasi film" />

## Evaluation

Pada tahap _evaluation_ akan dilakukan analisis terhadap sistem rekomendasi yang dibuat dengan menggunakan teknik _content-based filtering_ dan _collaborative filtering_.

### _Content-Based Filtering_
Evaluasi akan dilakukan dengan mencoba memasukkan _inputan_ sebuah _film_ berjudul "Avatar". Setelah itu sistem menampilkan list beberapa rekomendasi _film_ yang _relate_ dengan _film_ tersebut. Setelah melakukan analisis secara manual dengan mengecek _detail_ masing-masing dari _film_ tersebut sistem berhasil menampilkan rekomendasi _film_ yang relevan. Kemudian akan dihitung jumlah rekomendasi _film_ yang relevan menggunakan rumus _precision_. Hasilnya dari 3 _film_ sebanyak 3 _film_ menampilkan _film_ yang relevan dengan persentase sebesar 100%.

Berikut merupakan rumus dari _precision_:

<img width="400" src="https://github.com/danalvr/MLT-Assignment-Dicoding/assets/81479217/38bf2d9a-dec7-4121-95cd-983e628c10e1" alt="Hasil rekomendasi film" />

Berikut merupakan hasil rekomendasi _film_ yang relevan dengan _film_ "Avatar":

<img width="400" src="https://github.com/danalvr/MLT-Assignment-Dicoding/assets/81479217/b867ac85-a4ef-4010-922a-a7e116b4dec9" alt="Hasil rekomendasi film" />

### _Collaborative filtering_

Evaluasi akan dilakukan dengan menggunakan metrik RMSE pada saat melakukan _training_ model. Cara kerja metrik RMSE yaitu dengan menghitung selisih antara nilai aktual dan nilai prediksi. Kemudian setiap selisih tersebut akan dikuadratkan dan hitung rata-rata-nya yang kemudian dilakukan akar kuadrat dari hasil rata-rata tersebut. Semakin kecil nilai RMSE, semakin baik kinerja dari model sistem rekomendasi yang dibuat.

Berikut merupakan rumus RMSE:

<img width="400" src="https://github.com/danalvr/MLT-Assignment-Dicoding/assets/81479217/454904ca-65f8-4c92-b1ba-eea865ca2b56" alt="Metrik RMSE" />

Berikut merupakan hasil visualisasi data latih dan data uji menggunakan metrik RMSE:

<img width="400" src="https://github.com/danalvr/MLT-Assignment-Dicoding/assets/81479217/1a11b5f1-5a62-47be-be21-ca45c3fe4d28" alt="Metrik RMSE" />

Berdasarkan hasil analisis dari grafik diatas model mengalami konvergen setelah _epoch_ ke-8 dan nilai RMSE mendekati nol yang menandakan bahwa hasil _training_ dari model rekomendasi memiliki kinerja yang baik.
