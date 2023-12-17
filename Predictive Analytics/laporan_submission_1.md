# Laporan Proyek Machine Learning - Daniel Alvaro Sormin

## Domain Proyek

Di tengah pertumbuhan pasar properti yang cepat, pemahaman yang akurat tentang nilai properti menjadi kunci utama bagi pembeli, penjual, dan pemilik rumah. Dalam konteks ini, proyek machine learning prediksi harga rumah menjadi instrumen yang sangat bermanfaat untuk memberikan estimasi nilai properti berdasarkan sejumlah faktor.

Tujuan dari proyek ini adalah untuk mengembangkan model machine learning yang dapat memprediksi harga rumah berdasarkan fitur-fitur tertentu, seperti luas rumah, kamar tidur, perabotan, kedekatan dengan jalan raya dan karakteristik lainnya. Model ini diharapkan dapat memberikan estimasi harga yang lebih akurat dibandingkan dengan metode tradisional.

## Business Understanding

Pada tahapan business understanding terdiri dari 3 bagian yaitu problem statements, goals dan solution statements. Berikut merupakan rincian dari business understanding:

### Problem Statements

1. Ketidakpastian Harga Properti: Pembeli dan penjual sering menghadapi ketidakpastian dalam menentukan harga yang wajar untuk properti.

2. Keterbatasan Informasi: Kurangnya informasi yang akurat dan cepat tentang nilai properti dapat menghambat proses pengambilan keputusan.

3. Tren Pasar yang Dinamis: Pasar properti cenderung berubah dengan cepat, dan seringkali sulit untuk menyesuaikan harga properti dengan tren pasar yang dinamis.

### Goals

1. Menyediakan Estimasi Harga yang Akurat: Mengembangkan model machine learning yang dapat memberikan estimasi harga rumah yang akurat berdasarkan faktor-faktor tertentu.

2. Meningkatkan Keterbukaan Informasi: Memberikan platform yang memberikan akses cepat dan transparan terhadap informasi harga properti dengan menggunakan teknologi machine learning.

3. Peningkatan Efisiensi Pengambilan Keputusan: Membantu pembeli dan penjual dalam pengambilan keputusan yang lebih efisien dan informan dengan memberikan perkiraan harga yang handal.

### Solution statements

Dengan mengimplementasikan model machine learning untuk prediksi harga rumah, proyek ini bertujuan untuk memberikan solusi yang inovatif dan adaptif terhadap ketidakpastian dan keterbatasan informasi dalam pasar properti. Solusi ini akan menciptakan platform yang memanfaatkan teknologi untuk memberikan estimasi harga yang akurat, memfasilitasi proses pengambilan keputusan, dan meningkatkan keterbukaan informasi di pasar properti. Dengan demikian, proyek ini mendukung terciptanya pasar properti yang lebih efisien, transparan, dan responsif terhadap perubahan pasar.

## Data Understanding

Dataset yang digunakan pada project machine learning untuk mengukur prediksi harga rumah berasal dari [Kaggle](https://www.kaggle.com/datasets/yasserh/housing-prices-dataset). Data tersebut terdiri dari 13 fitur dan memiliki total sebanyak 545.

### Variabel-variabel pada House Pricing Dataset adalah sebagai berikut:

- price : Variabel target yang menyatakan harga rumah.
- area : Luas total tanah atau bangunan rumah.
- bedrooms : Jumlah kamar tidur dalam rumah.
- bathrooms : Jumlah kamar mandi dalam rumah.
- stories : Jumlah lantai atau tingkat bangunan rumah.
- mainroad : Variabel yang menunjukkan apakah rumah tersebut terletak di jalan utama atau tidak.
- guestroom : Variabel yang menunjukkan apakah rumah tersebut memiliki kamar tamu atau tidak.
- basement : Variabel yang menunjukkan apakah rumah tersebut memiliki ruang bawah tanah atau tidak.
- hotwaterheating : Variabel yang menunjukkan apakah rumah tersebut memiliki pemanas air atau tidak.
- airconditioning : Variabel yang menunjukkan apakah rumah tersebut dilengkapi dengan sistem pendingin udara atau tidak.
- parking : Jumlah tempat parkir yang tersedia di rumah.
- prefarea : Variabel yang menunjukkan apakah rumah tersebut terletak di daerah yang diinginkan atau tidak.
- furnishingstatus : Status perabotan rumah, seperti furnished (perabotan lengkap), semi-furnished (sebagian perabotan), atau unfurnished (tidak berperabotan).

### Teknik yang digunakan pada tahapan data understanding

- Melakukan Variabel Description : Untuk memeriksa setiap variabel yang terdapat dalam dataset apakah terdapat nilai NaN, duplicate, dll.
- Melakukan Teknik Univariarate Analysis : Untuk menganalisis setiap variabel secara terpisah dengan fokus pada distribusi nilai, statistik deskriptif, dan visualisasi.
- Melakukan Teknik Multivariate Analysis : Untuk memahami hubungan antara dua atau lebih variabel dalam dataset.

## Data Preparation

- Teknik Encoding : Melakukan teknik pengkodean pada fitur kategori untuk memberikan representasi yang lebih tepat dalam bentuk numerik, memungkinkan model machine learning memahami dan memproses informasi kategorikal.

- Teknik Principal Component Analysis (PCA) : Menerapkan teknik PCA untuk mereduksi dimensi fitur dalam dataset, mempertahankan sebagian besar varian sambil mengurangi kompleksitas model.

- Teknik Train-Test Split : Membagi dataset menjadi data latih dan data uji dengan perbandingan 9:1, di mana data latih digunakan untuk melatih model dan data uji digunakan untuk menguji kinerja model. Perbandingan tersebut dapat disesuaikan berdasarkan kebutuhan spesifik proyek.

## Modeling

Membangun model untuk menghitung prediksi harga rumah dengan membandingkan berbagai algoritma yaitu KNN, Random Forest dan Boosting untuk mencari hasil akurasi terbaik. Adapun kelebihan dan kekurangan dari algoritma yang digunakan sebagai berikut:

### Algoritma K-Nearest Neighbors (KNN):

Kelebihan:

- Sederhana dan Mudah Dipahami: KNN adalah algoritma yang relatif mudah dipahami dan diimplementasikan.
- Tidak Memerlukan Training: KNN termasuk dalam kategori algoritma lazy learning, yang berarti tidak memerlukan fase pelatihan yang panjang.
- Efektif pada Data dengan Struktur yang Rumit: KNN dapat efektif menangani dataset yang memiliki struktur yang kompleks atau tidak teratur.

Kekurangan:

- Sensitif terhadap Outlier: KNN dapat sangat dipengaruhi oleh keberadaan outlier dalam data.
- Mahal secara Komputasional: Perlu menghitung jarak dari setiap titik data terhadap semua titik data lainnya, sehingga bisa menjadi mahal secara komputasional, terutama pada dataset besar.
- Memerlukan Penanganan Data yang Baik: KNN memerlukan penanganan data yang baik, seperti normalisasi, karena rentan terhadap perbedaan skala antar fitur.

### Algoritma Random Forest:

Kelebihan:

- Akurasi yang Tinggi: Random Forest memiliki kinerja yang baik dan seringkali menghasilkan model yang akurat.
- Robust terhadap Overfitting: Kemampuannya untuk menggabungkan hasil dari banyak pohon keputusan membuatnya lebih tahan terhadap overfitting.
- Mampu Menangani Data yang Tidak Seimbang: Random Forest dapat mengatasi masalah ketidakseimbangan kelas dalam klasifikasi.

Kekurangan:

- Kompleksitas Model: Model Random Forest yang kompleks dapat sulit untuk diinterpretasi, terutama jika terdiri dari ratusan pohon keputusan.
- Memerlukan Banyak Waktu untuk Pelatihan: Proses pelatihan pada Random Forest bisa memakan waktu, terutama pada dataset besar.
- Memerlukan Banyak Memori: Karena modelnya terdiri dari banyak pohon, Random Forest membutuhkan lebih banyak memori.

### Algoritma Boosting:

Kelebihan:

- Mampu Mengatasi Bias (Boosting): Algoritma Boosting dapat mengurangi bias dan meningkatkan akurasi model.
- Efektif pada Dataset Besar: Meskipun lebih kompleks daripada KNN, algoritma Boosting dapat efektif bekerja pada dataset yang besar.
- Mengurangi Overfitting (AdaBoost): Beberapa implementasi Boosting, seperti AdaBoost, dapat membantu mengurangi overfitting.
  Kekurangan:

- Sensitif terhadap Noise dan Outlier: Algoritma Boosting dapat menjadi sensitif terhadap noise dan outlier dalam data.
- Memerlukan Penyesuaian Parameter yang Baik: Memilih parameter yang tepat untuk Boosting dapat menjadi tantangan dan memerlukan penyesuaian yang cermat.

Setelah melakukan training, model terbaik yang dapat digunakan yaitu model yang menggunakan algoritma KNN.

## Evaluation

Setelah melakukan training dilakukan evaluasi untuk mengukur kinerja model menggunakan metrik MSE. Seperti yang diketahui MSE merupakan salah satu metrik evaluasi yang umum digunakan dalam proyek prediksi. Metrik ini mengukur rata-rata dari selisih kuadrat antara nilai prediksi dan nilai sebenarnya dari suatu model. Semakin kecil nilai MSE, semakin baik modelnya. Nilai MSE yang rendah menunjukkan bahwa model cenderung memiliki prediksi yang dekat dengan nilai sebenarnya. MSE memberikan penalti yang besar untuk kesalahan yang besar, karena selisih di kuadrat sebelum dihitung rata-rata. Adapun hasil project setelah melakukan evaluasi adalah model yang dihasilkan berada dalam kondisi underfitting dimana nilai akurasi data training dan data testing yang rendah. Salah satu penyebab terjadi underfitting dikarenakan dataset yang digunakan masih terlalu sedikit.
