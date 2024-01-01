### Laporan Proyek Machine Learning

### Nama : Moch Reki Hadiyanto

### Nim : 211351083

### Kelas : IF Pagi B

## Domain Proyek
Web app yang bisa digunakan pada toko bakery prancis untuk memberikan rekomendasi produk tambahan kepada pelanggan yang membeli pada toko bakery tersebut.

## Business Understanding
Web app ini dikembangkan untuk mengeluarkan potensi sales sebuah toko bakery, namun logika & proses dari pembuatannya bisa digunakan pada semua jenis toko asalkan terdapat datasetsnya. Dengan memberikan rekomendasi produk lain pada pelanggan, maka toko bisa menghasilkan net profit yang lebih tinggi dan memberikan review positif pada toko karena telah memberikan rekomendasi terbaik bagi pelanggan yang kebingungan.

### Problem Statement
- Tidak keluarnya potensi penuh profit dari toko
- Pelanggan memiliki rasa indecisive yang tinggi sehingga kebingungan saat memesan dan memakan waktu yang lama saat memesan.
### Goals
- Menghasilkan net profit toko yang tinggi
- Membantu pelanggan memilih pasangan produk yang sesuai untuk produk yang dia beli dengan cepat.
### Solution Statements
- Membuatkan web app dengan algorithma apriori untuk memberikan rekomendasi pasangan produk yang tepat bagi pelanggan.

## Data Understanding
Dataset ini saya dapatkan dari kaggle.com. Yang mana dataset ini berasal dari toko bakery French. Ianya memberikan data tentang detail transaksi harian pelanggan dari 2021-01-01 hingga 2022-09-30. Dataset ini mengandung 234,005 baris data dan lebih dari 136,000 transaksi dan 6 kolom data.
Dataset = [French Bakery Daily Sales](https://www.kaggle.com/datasets/matthieugimbert/french-bakery-daily-sales)

### Variabel-variabel pada French Bakery Daily Sales adalah sebagai berikut:

- date : Menunjukkan tanggal orderan dibuat (date, 2021-01-02 hingga 2022-09-30)
- time : Menunjukkan waktu orderan dibuat (time)
- ticket_number : Menunjukkan tiap-tiap transaksi (int, 150,000 hingga 289,000)
- article : Menunjukkan menu yang dipilih/dipesan (object, ada banyak)
- quantity : Menunjukkan jumlah pembelian dari produk (int, 0 hingga 8)
- unit_price : Menunjukkan harga per produknya (float, 0.15 hingga 9.80)

## Data Preparation
Pada tahap ini saya akan melakukan data exploration serta data visualization. Lalu melakukan beberapa modifikasi data agar data-datanya memungkinkan untuk diproses oleh algorithma apriori.

### Import Dataset
Seperti biasa langkah pertama adalah memasukkan datasets yang sudah kita pilih.
```bash
from google.colab import files
files.upload()
```
```bash
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!ls ~/.kaggle
```

```bash
!kaggle datasets download -d matthieugimbert/french-bakery-daily-sales
```
Selesai mengunduh datasets, mari lanjut dengan mengextract filenya.
```bash
!unzip french-bakery-daily-sales.zip -d dataset
!ls dataset 
```
### Import library 
```bash
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import association_rules, apriori
import plotly.express as px
import re
import pickle
```

### Import Dataset pada Varible
Kita memasukkan file csv yang tadi telah diextract pada sebuah variable yang bernama df.
```bash
df = pd.read_csv('dataset/Bakery sales.csv')
```
Melihat 5 data awal dari datasets.
```bash
df.head()
```
Kita bisa lihat dibawah mayoritas datatype dari datasetsnya adalah object.
``` bash
df.info()
```
```bash
df.describe()
```
Diatas menunjakkan nilai mean, min hingga max dari kolom-kolom yang numeric.
```bash
df.isnull().sum()
```
```bash
df['article'].sort_values().unique()
```
Wow, diatas merupakan nilai unique dari kolom article, yang artinya itu merupakan menu yang tersedia pada toko french ini. Cukup banyak ya.
```bash
df.shape
```
kita bisa lihat bahwa datasets ini memiliki 234,005 jumlah baris data dengan 7 kolom.
### Data Cleansing
```bash
df["date"] = pd.to_datetime(df["date"] + " " + df["time"])
df.rename(columns={"date":"date_time"},inplace=True)
```
Code diatas digunakan untuk menyatukan kolom data & time lalu mengubah namanya menjadi kolom date_time.
Sedangkan code dibawah itu untuk memisahkan nilai year, month, day, dan dayofweek pada kolomnya sendiri.
```bash
df["Y"] = df["date_time"].dt.year
df["M"] = df["date_time"].dt.month
df["D"] = df["date_time"].dt.day
df["WD"] = df["date_time"].dt.dayofweek + 1
```
Selanjutnya adalah mengubah kolom Quantity dengan datatype integer. Dan menghilangkan simbol euro serta mengubah ',' menjadi '.' lalu mengubah datatypenya menjadi float agar mudah untuk diproses nantinya.
```bash
df.Quantity = df.Quantity.astype(int)
df.unit_price = df.unit_price.str.replace(' €', '').str.replace(',','.').astype(float)
```
Kita bisa mendapatkan nilai Revenue(keuntungan) dengan mengkalikan kolom Quantity dengan unit_price.
```bash
df["REV"] = df.Quantity * df.unit_price
```
Kita akan melakukan groupby perbulan untuk nanti melihat volume penjualan, jumlah client, dan revenue yang dihasilkan dibulan itu.
```bash
monthly = df.groupby("M", as_index=False).agg(total_sales_volume=("Quantity","sum"), client_num=("ticket_number","count"),total_rev=("REV","sum"))
monthly.loc[:8,monthly.columns.difference(["M"])] = monthly.loc[:8,monthly.columns.difference(["M"])].applymap(lambda x:x/2)
```
Selanjutnya kita akan melakukan groupby per hari, sama seperti diatas, kita akan melihat volume Quantity, jumlah client dan total revenue yang dihasilkan.
```bash
daily = df.groupby("D", as_index=False).agg(total_sales_volume=("Quantity","sum"), client_num=("ticket_number","count"),total_rev=("REV","sum"))
```
Kita akan melakukan groupby perminggu juga.
```bash
weekly = df.groupby("WD", as_index=False).agg(total_sales_volume=("Quantity","sum"), client_num=("ticket_number","count"),total_rev=("REV","sum"))
```
dan terakhir kita akan melakukan groupby perjam juga.
```bash
df["hr"] = df["date_time"].dt.hour
hourly = df.groupby("hr", as_index=False).agg(total_sales_volume=("Quantity","sum"), client_num=("ticket_number","count"),total_rev=("REV","sum"))
```
## Visualisasi
Mari lanjut dengan menunjukkan hasil groupby-groupby diatas untuk menganalisa sales dan revenuenya.
```bash
fig = px.line(monthly,x="M",y=["client_num","total_sales_volume","total_rev"],title="Monthly number of clients, total sales volume and total revenue")
fig.show()
```
![newplot](https://github.com/RexBaee/product-recommendation/assets/130348460/7942027f-5e0b-4b62-8289-0b7f4e8ae77b) <br>
Nah, diatas menunjukkan ketiga data yang tadi telah dikelompokkan berdasarkan bulan. Kita bisa lihat bahwa semuanya mengelonjak tinggi dari bulan 6 dengan bulan ke 8 sebagai puncaknya. Itu merupakan bulan penting bagi toko. <br>
Selanjutnya kita akan melihat jumlah revenue yang dihasilkan masing-masing bulan dalam setahun. (based on percentage tentunya)
```bash
fig = px.pie(monthly,names="M",values="total_rev",hole=0.5,title="How much revenue does each month account for?")
fig.update_traces(textposition='outside',textinfo='percent+label',sort=False)
fig.show()
```
![newplot (1)](https://github.com/RexBaee/product-recommendation/assets/130348460/661c1d8b-0e14-4adf-8049-cb6f25eec9a8)
Nah, betul saja bulan ke 8 menghasilkan 16.6% revenue dari tahun ini, diikuti dengan bulan ke 7 dan dilanjut dengan bulan ke 5. <br>
Selanjutnya kita akan melihat based on day(daily), sama seperti diatas, kita akan melihat volume Quantity, jumlah client dan total revenue yang dihasilkan.
```bash
fig = px.line(daily,x="D",y=["client_num","total_sales_volume","total_rev"],title="Daily number of clients, total sales volume and total revenue")
fig.show()
```
![newplot (2)](https://github.com/RexBaee/product-recommendation/assets/130348460/ce5eb57e-4c40-4ba5-8db9-a149e0b93d62)
Penjualan meningkat pada pertengahan bulan yaitu tanggal 13 hingga 15 dengan puncak sekitar tanggal 14. Kita akan lihat dalam seminggu itu dihari keberapakah penjualan meningkat secara pesat.
```bash
fig = px.line(weekly,x="WD",y=["client_num","total_sales_volume","total_rev"],title="Number of clients, total sales volume and total revenue inside a week")
fig.show()
```
![newplot (3)](https://github.com/RexBaee/product-recommendation/assets/130348460/6f0898d1-de66-48a3-992b-afa7ae5a0698)
Ya!, saya tidak terkejut karena weekend memanglah hari yang tepat untuk menikmati makanan manis, maka dari itu dihari-hari weekendlah penjualan selalu meningkat. <br>
Kita akan melihat dengan lebih merinci lagi dengan melihat tiap jamnya dalam sehari, kita akan melihat hal yang sama yaitu jumlah client, volume quantity penjualan dan total revenue.
```bash
fig = px.line(hourly,x="hr",y=["client_num","total_sales_volume","total_rev"],title="number of clients, total sales volume and total revenue inside a day")
fig.show()
```
![newplot (4)](https://github.com/RexBaee/product-recommendation/assets/130348460/587e17e6-7839-45a9-a9d8-b02e2584014f) <br>
Hasil yang tidak mengejutkan juga, penjualan meningkat pada jam-jam istirahat.
## Modeling
### Preprocessing
Kita akan melanjutkan tahap Preprocessing di tahap yang sebenarnya.
```bash
transactions = df['ticket_number'].nunique()
```
Kita memasukkan ticket_number(number penjualan) pada variable transaction dan terdapat 136451 transaksi pada datasets.
```bash
df.loc[(df['article'] == 'REDUCTION SUCREES 24') | (df['article'] == 'REDUCTION SUCREES 12')]
```
```bash
df['article_cat'] = df['article'].apply(lambda x: re.sub('[0-9](.*)','',x))\
.str.strip()
```
Selanjutnya kita akan menghilangkan beberapa nilai yang terdapat pada kolom article, lalu memasukkannya pada kolom article_cat.
```bash
delete_list = ['', ' ', '.', 'ARTICLE', 'THE']

df = df[df['article_cat'].isin(delete_list) == False]
```
```bash
sales_by_ticket = pd.DataFrame(df.groupby(['ticket_number', 'article_cat'])['Quantity'].count()).reset_index()
```
Membuat variable untuk menampung jumlah penjualan berdasarkan number ticket, jadi jika number ticketsnya sama maka itu termasuk dalam satu sales.
```bash
sales_grouped = pd.crosstab(index=sales_by_ticket.ticket_number, columns=sales_by_ticket.article_cat)\
.rename_axis(None)
```
```bash
sales_grouped = sales_grouped.applymap(lambda x: True if x>= 1 else False)
```
```bash
sales_grouped.head()
```
Kita bisa lihat sekarang tiap tiap menunya memiliki nilai true (jika ada pada pembelian) dan nilai false (jika tidak ada).

### Modeling
Kita akan lanjut dengan tahap modeling, disini saya akan menggunakan metric support untuk membuat modelnya karena data yang tersedia dan yang akan dihasilkan akan sangat banyak sekali.
```bash
frequent_itemsets = apriori(sales_grouped, min_support=0.005,use_colnames=True, max_len=4)
```
```bash
rules = association_rules(frequent_itemsets, metric="support",min_threshold=0.005)
rules.head(15)
```
### Visualisasi hasil algoritma
Dibawah menunjukkan bahwa COUPE merupakan produk yang paling sering dibeli dengan 83% confidence akan selalu memiliki pasangan pembelian, seperti COUPE dan BOULE sangat sering muncul pada pembelian.
```bash
rules.sort_values(['confidence','lift','support'],ascending=[False, False, True]).head(15)
```
## Evaluation
Berdasar table yang dihasilkan oleh apriori assosiation ini, produk yang paling populer adalah COUPE dan terdapat banyak sekali pasangan produk dengan COUPE ini. Confidencenya yang tinggi juga sangat mudah untuk dipercayai dan sangat menyakinkan bahwa produk itu adalah produk yang populer dan rekomendasi-rekomendasi yang diberikan juga sudah kombinasi yang terbaik.
## Deployment
[Aplikasi Rekomendasi Produk](https://app-recommendation-reki.streamlit.app/)
![deployed](https://github.com/RexBaee/product-recommendation/assets/130348460/fa1bbb6d-63dd-46fa-b5fc-d35d6423f71f)

