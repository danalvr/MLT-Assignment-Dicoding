# Laporan Proyek Machine Learning - Daniel Alvaro Sormin

## Domain Proyek

Di tengah pertumbuhan pasar properti yang dinamis, pemahaman yang akurat tentang nilai properti menjadi kunci utama bagi pembeli, penjual, dan pemilik rumah. Dalam konteks ini, proyek *machine learning* prediksi harga rumah menjadi instrumen yang sangat bermanfaat untuk memberikan estimasi nilai properti berdasarkan sejumlah faktor. Keberhasilan proyek ini akan memberikan dampak positif yang signifikan dalam konteks pertumbuhan pasar properti.

Proyek ini bertujuan untuk mengembangkan model *machine learning* yang dapat memprediksi harga rumah dengan presisi tinggi, mendasarkan diri pada fitur-fitur seperti luas rumah, jumlah kamar tidur, perabotan, kedekatan dengan jalan raya, dan karakteristik lainnya. Dengan melakukan hal ini, proyek ini memberikan solusi konkret dengan memberikan alat yang dapat memberikan estimasi harga yang lebih akurat dibandingkan metode tradisional.

Keuntungan konkret dari proyek ini mencakup memberikan pandangan yang lebih mendalam dan terperinci tentang nilai properti, yang dapat digunakan oleh pembeli untuk membuat keputusan pembelian yang informasional, penjual untuk menilai dengan tepat propertinya, dan pemilik rumah untuk merencanakan investasi atau penjualan mereka secara lebih efektif. Dengan mengintegrasikan teknologi *machine learning*, proyek ini memberikan solusi yang relevan dan memberdayakan para pemangku kepentingan di pasar properti untuk membuat keputusan yang lebih cerdas.

## Business Understanding

Proyek ini dibangung agar memberikan gambaran kepada individu/perusahaan yang ingin melakukan transaksi jual beli rumah.

### Problem Statements

1. Ketidakpastian Harga Properti: Pembeli dan penjual sering menghadapi ketidakpastian dalam menentukan harga yang wajar untuk properti. Contoh Kasus: Seorang penjual mengalami kesulitan menentukan harga yang sesuai untuk rumahnya karena fluktuasi pasar dan variasi harga properti di sekitarnya. Sebaliknya, seorang pembeli merasa khawatir membayar lebih dari nilai sebenarnya karena ketidakpastian dalam menilai properti.

2. Keterbatasan Informasi: Kurangnya informasi yang akurat dan cepat tentang nilai properti dapat menghambat proses pengambilan keputusan. Contoh Kasus: Seorang calon pembeli kesulitan mendapatkan informasi yang akurat dan lengkap tentang kondisi properti yang akan dibelinya. Informasi yang kurang dapat menyebabkan keputusan yang kurang optimal, terutama dalam hal perawatan dan biaya tambahan yang mungkin diperlukan setelah pembelian.

3. Tren Pasar yang Dinamis: Pasar properti cenderung berubah dengan cepat, dan seringkali sulit untuk menyesuaikan harga properti dengan tren pasar yang dinamis. Contoh Kasus: Seorang penjual yang tidak menyadari tren pasar terkini mungkin menetapkan harga properti di luar kisaran wajar, yang dapat mengakibatkan propertinya sulit terjual. Di sisi lain, seorang pembeli yang tidak memiliki akses informasi terbaru mungkin membayar lebih dari yang seharusnya karena ketidakpahaman terhadap tren pasar yang dinamis.

### Goals

1. Menyediakan Estimasi Harga yang Akurat: Mengembangkan model *machine learning* yang dapat memberikan estimasi harga rumah yang akurat berdasarkan faktor-faktor tertentu. Metrik Keberhasilan: Mencapai tingkat akurasi prediksi harga rumah di atas 90% berdasarkan perbandingan antara harga aktual dan harga yang diprediksi oleh model *machine learning*.

2. Meningkatkan Keterbukaan Informasi: Memberikan platform yang memberikan akses cepat dan transparan terhadap informasi harga properti dengan menggunakan teknologi *machine learning*. Metrik Keberhasilan: Mempercepat akses informasi dengan menurunkan waktu rata-rata yang dibutuhkan oleh pengguna untuk mendapatkan data harga properti, dinilai melalui pengukuran kecepatan respon platform.

3. Peningkatan Efisiensi Pengambilan Keputusan: Membantu pembeli dan penjual dalam pengambilan keputusan yang lebih efisien dan informan dengan memberikan perkiraan harga yang handal. Metrik Keberhasilan: Meningkatkan efisiensi pengambilan keputusan, diukur dengan pengurangan waktu yang dibutuhkan oleh pembeli dan penjual untuk menetapkan harga atau membuat keputusan berdasarkan estimasi yang diberikan oleh model.

### Solution statements

Dengan menerapkan model machine learning untuk memprediksi harga rumah, proyek ini bertujuan untuk menghadirkan solusi yang inovatif dan adaptif terhadap ketidakpastian serta keterbatasan informasi di pasar properti. Solusi ini akan menciptakan platform yang memanfaatkan teknologi untuk memberikan estimasi harga yang akurat, memfasilitasi proses pengambilan keputusan, dan meningkatkan keterbukaan informasi di pasar properti.

Secara spesifik, platform ini akan memberikan akses cepat dan transparan terhadap informasi harga properti dengan menyajikan data aktual dan estimasi harga secara langsung kepada pengguna. Pengguna dapat dengan mudah mengakses dan membandingkan harga properti di berbagai lokasi, memungkinkan mereka membuat keputusan yang lebih terinformasi. Selain itu, fitur visualisasi data yang intuitif akan membantu pengguna dalam memahami tren harga properti secara lebih baik. Dengan demikian, platform ini tidak hanya menyederhanakan akses informasi tetapi juga memberikan pengalaman yang lebih informatif dan efektif bagi para pemangku kepentingan di pasar properti.

## Data Understanding

Dataset yang digunakan pada project machine learning untuk mengukur prediksi harga rumah berasal dari [House Pricing - Kaggle](https://www.kaggle.com/datasets/yasserh/housing-prices-dataset). Data tersebut terdiri dari 13 fitur dan memiliki total sebanyak 545 entri. Data tersebut dikumpulkan menggunakan metode scrapping pada google search.

### Variabel-variabel pada dataset:

- _price_ : Variabel target yang menyatakan harga rumah.
- _area_ : Luas total tanah atau bangunan rumah.
- _bedrooms_ : Jumlah kamar tidur dalam rumah.
- _bathrooms_ : Jumlah kamar mandi dalam rumah.
- _stories_ : Jumlah lantai atau tingkat bangunan rumah.
- _mainroad_ : Variabel yang menunjukkan apakah rumah tersebut terletak di jalan utama atau tidak.
- _guestroom_ : Variabel yang menunjukkan apakah rumah tersebut memiliki kamar tamu atau tidak.
- _basement_ : Variabel yang menunjukkan apakah rumah tersebut memiliki ruang bawah tanah atau tidak.
- _hotwaterheating_ : Variabel yang menunjukkan apakah rumah tersebut memiliki pemanas air atau tidak.
- _airconditioning_ : Variabel yang menunjukkan apakah rumah tersebut dilengkapi dengan sistem pendingin udara atau tidak.
- _parking_ : Jumlah tempat parkir yang tersedia di rumah.
- _prefarea_ : Variabel yang menunjukkan apakah rumah tersebut terletak di daerah yang diinginkan atau tidak.
- _furnishingstatus_ : Status perabotan rumah, seperti _furnished_ (perabotan lengkap), semi-_furnished_ (sebagian perabotan), atau _unfurnished_ (tidak berperabotan).

### Teknik yang digunakan pada tahapan data understanding

- Variabel _Description_ : Untuk memeriksa setiap variabel yang terdapat dalam dataset apakah terdapat nilai _NaN_, _duplicate_, dll.
- _Univariarate Analysis_ : Untuk menganalisis setiap variabel secara terpisah dengan fokus pada distribusi nilai, statistik deskriptif, dan visualisasi.
- _Multivariate Analysis_ : Untuk memahami hubungan antara dua atau lebih variabel dalam dataset. Hasil analisis multivariate dapat mencakup identifikasi korelasi antara luas rumah dengan harga, pengaruh jumlah kamar mandi terhadap harga, dan sejenisnya. Hasil analisis ini akan membantu pemilihan fitur yang optimal untuk model prediksi harga rumah. Berikut merupakan contoh hasil analisis _multivariate_:
<img width="400" src="https://github.com/danalvr/MLT-Assignment-Dicoding/assets/81479217/a0df5e26-6144-4a98-8c38-aed7f6fe5080" alt="gambar korelasi matrix" />

Pada gambar diatas dapat disimpulkan bahwa _area_, _bathrooms_ dan _stories_ sangat berpengaruh dalam menentukan suatu harga.

## Data Preparation

- Teknik _Encoding_ : Melakukan teknik *encoding* pada fitur kategori bertujuan untuk memberikan representasi yang lebih akurat dalam bentuk numerik. Ini esensial karena model machine learning bekerja lebih efektif dengan data dalam bentuk numerik. Pengkodean ini memungkinkan model memahami dan memproses informasi kategorikal, seperti status perabotan rumah atau lokasi pada prefered area, dengan lebih baik.

- Teknik _Principal Component Analysis_ (PCA) : Penerapan teknik PCA bertujuan untuk mereduksi dimensi fitur dalam dataset. Dengan mempertahankan sebagian besar varian data, PCA membantu mengurangi kompleksitas model. Hal ini bermanfaat untuk mempercepat waktu pelatihan, dan meningkatkan generalisasi model pada data baru.

- Teknik _Train-Test Split_ : Membagi dataset menjadi data latih dan data uji dengan perbandingan 8:2 dimana data latih digunakan untuk melatih model dan data uji digunakan untuk menguji kinerja model. Rasio 8:2 digunakan karena dataset memilki jumlah yang sedikit sehingga dibutuhkan data latih yang banyak dan data uji yang tidak terlalu sedikit. Hal tersebut telah dibuktikan dengan membandingkan hasil pengujian dari model dengan rasio 7:3 dan 9:1 dengan menggunakan metrik MSE. Hasil pengujian dengan rasio berbeda akan dilampirkan pada bagian _evaluation_.

- Teknik _Standardization_ : Melakukan _standardization_ atau normalisasi dataset agar algoritma _machine learning_ yang digunakan memiliki performa yang lebih baik dan bekerja lebih cepat. Normalisasi bekerja dengan cara menyeragamkan dataset agar memilki skala yang relatif sama. Adapun teknik yang digunakan dalam normalisasi dataset yaitu dengan metode _StandardScaler_. Dengan menggunakan _StandardScaler_, setiap fitur numerik diubah sedemikian rupa sehingga memiliki rata-rata nol dan deviasi standar satu.

## Modeling

Membangun model untuk menghitung prediksi harga rumah dengan membandingkan berbagai algoritma yaitu KNN, Random Forest dan Boosting untuk mencari hasil akurasi terbaik dengan menggunakan metrik MSE (Mean Squared Error). Adapun kelebihan dan kekurangan dari algoritma yang digunakan sebagai berikut:

### Algoritma K-Nearest Neighbors (KNN):

Kelebihan:

- Sederhana dan Mudah Dipahami: KNN adalah algoritma yang relatif mudah dipahami dan diimplementasikan.
- Tidak Memerlukan Training: KNN termasuk dalam kategori algoritma _lazy learning_, yang berarti tidak memerlukan fase pelatihan yang panjang.
- Efektif pada Data dengan Struktur yang Rumit: KNN dapat efektif menangani dataset yang memiliki struktur yang kompleks atau tidak teratur.

Kekurangan:

- Sensitif terhadap Outlier: KNN dapat sangat dipengaruhi oleh keberadaan _outlier_ dalam data.
- Mahal secara Komputasional: Perlu menghitung jarak dari setiap titik data terhadap semua titik data lainnya, sehingga bisa menjadi mahal secara komputasional, terutama pada dataset besar.
- Memerlukan Penanganan Data yang Baik: KNN memerlukan penanganan data yang baik, seperti normalisasi, karena rentan terhadap perbedaan skala antar fitur.

Implementasi:

Pertama, dilakukan inisialisasi model KNN Regressor dengan menyertakan parameter penting, seperti _n_neighbors_ yang menentukan jumlah tetangga terdekat yang akan dipertimbangkan dalam prediksi. Setelah inisialisasi, model dilatih pada data latih (_X_train_ dan _y_train_) dengan memanggil metode fit.

Proses pelatihan KNN tidak melibatkan pembentukan model yang kompleks seperti pada beberapa algoritma lainnya. Sebaliknya, selama tahap prediksi, algoritma menghitung jarak antara _instance_ yang akan diprediksi dan semua _instance_ dalam data latih. _Instance-instance_ ini kemudian diurutkan berdasarkan jarak, dan _k-neighbors_ terdekat dipilih.

Prediksi akhir untuk _instance_ yang diberikan adalah agregasi dari nilai target tetangga-tetangga ini. Pada regresi, ini sering kali merupakan rata-rata dari nilai target tetangga.

Setelah pelatihan, selanjutnya mengukur kinerja model pada data latih menggunakan Mean Squared Error (MSE). MSE memberikan gambaran tentang seberapa baik model dapat memprediksi nilai target dengan membandingkan prediksi dengan nilai sebenarnya. Hasil evaluasi ini kemudian disimpan dalam struktur data model, khususnya dalam variabel _train_mse_.

### Algoritma Random Forest:

Kelebihan:

- Akurasi yang Tinggi: Random Forest memiliki kinerja yang baik dan seringkali menghasilkan model yang akurat.
- Robust terhadap _Overfitting_: Kemampuannya untuk menggabungkan hasil dari banyak pohon keputusan membuatnya lebih tahan terhadap _overfitting_.
- Mampu Menangani Data yang Tidak Seimbang: Random Forest dapat mengatasi masalah ketidakseimbangan kelas dalam klasifikasi.

Kekurangan:

- Kompleksitas Model: Model Random Forest yang kompleks dapat sulit untuk diinterpretasi, terutama jika terdiri dari ratusan pohon keputusan.
- Memerlukan Banyak Waktu untuk Pelatihan: Proses pelatihan pada Random Forest bisa memakan waktu, terutama pada dataset besar.
- Memerlukan Banyak Memori: Karena modelnya terdiri dari banyak pohon, Random Forest membutuhkan lebih banyak memori.

Implementasi:

Pertama, dilakukan inisialisasi model RandomForestRegressor dengan beberapa parameter seperti _n_estimators_ (jumlah pohon), _max_depth_ (kedalaman maksimum setiap pohon), dan _random_state_ (untuk reproduktibilitas). Setelah inisialisasi, model dilatih menggunakan data latih (_X_train_ dan _y_train_) dengan memanggil metode _fit_.

Selama proses pelatihan, Random Forest membuat sejumlah pohon keputusan yang diterapkan pada subset acak dari data latih. Setiap pohon menghasilkan prediksi, dan prediksi dari seluruh pohon digunakan untuk menghasilkan prediksi akhir model. Pemilihan acak fitur pada setiap pembuatan pohon dan agregasi prediksi dari berbagai pohon membantu model ini dalam mengatasi _overfitting_ dan meningkatkan generalisasi.

Setelah pelatihan selesai, selanjutnya mengukur kinerja model menggunakan metrik evaluasi Mean Squared Error (MSE) pada data latih. MSE mengukur seberapa baik model dapat memprediksi nilai target dengan membandingkan prediksi dengan nilai sebenarnya.

Terakhir, hasil evaluasi dalam bentuk MSE pada data latih disimpan ke dalam struktur data models. Va­riabel train_mse menyimpan nilai MSE yang mencerminkan sejauh mana model sesuai dengan data latih.

### Algoritma Boosting:

Kelebihan:

- Mampu Mengatasi Bias (_Boosting_): Algoritma Boosting dapat mengurangi bias dan meningkatkan akurasi model.
- Efektif pada Dataset Besar: Meskipun lebih kompleks daripada KNN, algoritma Boosting dapat efektif bekerja pada dataset yang besar.
- Mengurangi Overfitting (AdaBoost): Beberapa implementasi Boosting, seperti AdaBoost, dapat membantu mengurangi _overfitting_.

Kekurangan:

- Sensitif terhadap _Noise_ dan _Outlier_: Algoritma Boosting dapat menjadi sensitif terhadap _noise_ dan _outlier_ dalam data.
- Memerlukan Penyesuaian Parameter yang Baik: Memilih parameter yang tepat untuk Boosting dapat menjadi tantangan dan memerlukan penyesuaian yang cermat.

Implementasi:

Pertama, dilakukan inisialisasi model AdaBoost Regressor dengan menyertakan parameter seperti _learning_rate_, yang mengontrol kontribusi dari setiap model lemah. Selanjutnya, model dilatih pada data latih (_X_train_ dan _y_train_) menggunakan metode _fit_.

Proses pelatihan terdiri dari serangkaian iterasi di mana model lemah (dalam konteks ini, Regressor) dibangun. Pada setiap iterasi, bobot diberikan pada setiap _instance_ berdasarkan seberapa baik model sebelumnya dapat memprediksi _instance_ tersebut. Model lemah baru kemudian dilatih pada data dengan bobot ini.

Setelah iterasi selesai, hasil prediksi dari semua model lemah digabungkan dengan memberikan bobot sesuai dengan kontribusi masing-masing model. Prediksi akhir untuk suatu _instance_ adalah hasil penjumlahan dari prediksi semua model lemah, dikalikan dengan bobot masing-masing model.

Setelah melakukan _training_, model terbaik yang dapat digunakan yaitu model yang menggunakan algoritma KNN.

## Evaluation

Setelah melakukan *training* dilakukan evaluasi untuk mengukur kinerja model menggunakan metrik MSE. Seperti yang diketahui MSE merupakan salah satu metrik evaluasi yang umum digunakan dalam proyek prediksi. Metrik ini mengukur rata-rata dari selisih kuadrat antara nilai prediksi dan nilai sebenarnya dari suatu model.

Rumus MSE:

MSE = $\frac{1}{n} \Sigma_{i=1}^n({y}-\hat{y})^2$

Keterangan:
- *n* adalah jumlah sampel atau observasi
- *yᵢ* adalah nilai aktual dari observasi ke-*i*
- *ȳ* adalah nilai rata-rata dari semua nilai aktual
- Simbol *Σ* menunjukkan penjumlahan dari *i* = 1 hingga *n*


Semakin kecil nilai MSE, semakin baik modelnya. Nilai MSE yang rendah menunjukkan bahwa model cenderung memiliki prediksi yang dekat dengan nilai sebenarnya. MSE memberikan penalti yang besar untuk kesalahan yang besar, karena selisih di kuadrat sebelum dihitung rata-rata.

Berikut hasil visualisasi pengujian dengan rasio berbeda:


_Training Loss_ dan _Test Loss_ dengan metrik MSE rasio 7:3 
<img width="400" src="https://github.com/danalvr/MLT-Assignment-Dicoding/assets/81479217/f3bd0e78-f991-4e13-9bac-653311a0ada7" alt="gambar korelasi matrix" />


_Training Loss_ dan _Test Loss_ dengan metrik MSE rasio 8:2 
<img width="400" src="https://github.com/danalvr/MLT-Assignment-Dicoding/assets/81479217/ccc12d5b-42a9-42ac-9753-7b0626caccaf" alt="gambar korelasi matrix" />


_Training Loss_ dan _Test Loss_ dengan metrik MSE rasio 9:1 
<img width="400" src="https://github.com/danalvr/MLT-Assignment-Dicoding/assets/81479217/2a251a20-1b91-4466-84b3-e785a539484f" alt="gambar korelasi matrix" />

Berdasarkan hasil pengujian tersebut maka rasio yang digunakan dalam pembagian data latih dan data uji adalah 8:2 sehingga hasil evaluasi model dan prediksi menggunakan rasio tersebut.


Berikut merupakan hasil evaluasi:
|          | train             | test              |
|----------|-------------------|-------------------|
| KNN      | 1209061703.782569 | 1903827800.545871 |
| RF       | 253313149.820106  | 2004744776.629099 |
| Boosting | 1357712627.128917 | 2293041311.803269 |

Model KNN memiliki MSE yang tinggi pada kedua set data, menunjukkan bahwa model ini cenderung memiliki kesalahan prediksi yang signifikan. Terdapat perbedaan yang cukup besar antara MSE pada data latih dan uji, yang bisa mengindikasikan adanya _overfitting_. 

Model Random Forest menunjukkan hasil yang baik pada data latih, dengan MSE yang lebih rendah dibandingkan dengan KNN. Namun, pada data uji, MSE meningkat, yang dapat mengindikasikan _overfitting_ atau kesulitan model dalam menggeneralisasi pola dari data yang belum pernah dilihat. 

Model Boosting memiliki MSE yang cukup tinggi pada kedua set data. Seperti KNN, ada perbedaan yang signifikan antara MSE pada data latih dan uji, yang dapat menunjukkan masalah _overfitting_. Model ini mungkin kesulitan dalam menggeneralisasi pola dari data yang baru.

Hasil Prediksi:

|     | y_true  | prediksi_KNN | prediksi_RF | prediksi_Boosting |
|-----|---------|--------------|-------------|-------------------|
| 526 | 2310000 | 2726500.0    | 2896553.3   | 3108727.3         |

Berdasarkan perbandingan antara nilai sebenarnya (_y_true_) dengan nilai yang diprediksi oleh model KNN, Random Forest, dan Boosting untuk suatu observasi tertentu (baris 526), terlihat bahwa model KNN memiliki prediksi yang paling mendekati nilai sebenarnya dengan nilai prediksi sebesar 2.726.500, diikuti oleh model Random Forest dengan prediksi sebesar 2.896.553,3, dan model Boosting dengan prediksi sebesar 3.108.727,3. Kesimpulannya, model KNN memberikan prediksi yang lebih akurat untuk kasus ini dibandingkan dengan model Random Forest dan Boosting.

