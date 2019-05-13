Inisiasi Virtual Environment

1. Masuk ke direktori root dari projek
2. Kalo belum install package virtualenv secara global, install dulu pakai command:
   pip install virtualenv
3. Jalankan command di bawah ini untuk membuat virtual environment:
   virtualenv env
4. Aktifkan virtual environment pakai command:
   venv\Scripts\activate.bat

Install package sesuai dengan yang ada di requirement.txt

1. pip install -r requirements.txt 
2. Semua package terinstall

Visualisasi Hasil Running Proses Training Menggunakan Tensorboard

1. Buka command line dan masuk ke direktori root dari projek
2. tensorboard --logdir logs
3. Akses localhost:6006




   