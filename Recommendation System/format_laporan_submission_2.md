# Laporan Proyek Machine Learning - Daniel Alvaro Sormin

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

  Berikut merupakan hasil analisis numerical:

Kemudian setelah melakukan analisis data menggunakan metode univariate analysis akan dilakukan tahapan data preprocessing dimana dataset credit.csv dan movie.csv akan digabung berdasarkan parameter movie_id pada dataset credit.

## Data Preparation

Pada tahap preparation dilakukan pengecekan dataset yang telah digabung sebelumnya. Setelah dilakukan pengecekan terhadap dataset terdapat beberapa missing value. Kemudian, missing value tersebut diganti dengan nilai string kosong.

### Demographic Filtering:

Pada bagian demographic filtering akan dilakukan pemetaan terhadap dataset dengan membuat sebuah fitur baru bernama score. Fitur score tersebut akan digunakan sebagai parameter sistem rekomendasi yang akan dikalkulasikan untuk melihat kesamaan data film pada teknik content-based filtering.

Gambar rumus penjelasan fitur score:

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

## Evaluation

Pada bagian ini Anda perlu menyebutkan metrik evaluasi yang digunakan. Kemudian, jelaskan hasil proyek berdasarkan metrik evaluasi tersebut.

Ingatlah, metrik evaluasi yang digunakan harus sesuai dengan konteks data, problem statement, dan solusi yang diinginkan.

**Rubrik/Kriteria Tambahan (Opsional)**:

- Menjelaskan formula metrik dan bagaimana metrik tersebut bekerja.

**---Ini adalah bagian akhir laporan---**

_Catatan:_

- _Anda dapat menambahkan gambar, kode, atau tabel ke dalam laporan jika diperlukan. Temukan caranya pada contoh dokumen markdown di situs editor [Dillinger](https://dillinger.io/), [Github Guides: Mastering markdown](https://guides.github.com/features/mastering-markdown/), atau sumber lain di internet. Semangat!_
- Jika terdapat penjelasan yang harus menyertakan code snippet, tuliskan dengan sewajarnya. Tidak perlu menuliskan keseluruhan kode project, cukup bagian yang ingin dijelaskan saja.
