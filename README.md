app
│
├── checkpoint/ --folder petrained model
│ ├── pretrained_classifier.pt --pengklasifikasi terlatih
│ └── pretrained_tfidf.pkl --korpus vektor tf-idf
│
├── models/ --folder model basis data
│ ├── history.py --class entitas riwayat
│ ├── lecturer.py --class entitas dosen
│ └── user.py --class entitas pengguna
│
├── resources/ --folder berkas kelengkapan
│
├── static/ --folder asset
│ ├── css/ --folder css
│ ├── images/ --folder gambar
│ ├── js/ --folder javascript
│ ├── manifest.json --konfigurasi pwa
│ └── service-worker.js --service pwa
│
├── templates/ --folder view antarmuka
│ ├── auth/ --folder antarmuka authentifikasi
│ ├── dashboard/ --folder antarmuka dashboard
│ └── error.html --view http error handling
│
├── views/ --folder controller (kelola view)
│ ├── **init** --bootstraping seluruh controller
│ ├── auth.py --controller authentifikasi
│ ├── classifier.py --controller klasifikasi
│ ├── dashboard.py --controller menu dashboard
│ ├── file.py --controller ekspor berkas
│ ├── history.py --controller riwayat
│ ├── lecturer.py --controller dosen
│ └── user.py --controller pengguna
│
├── .env --konfigurasi app
├── app.py --bootstraping seluruh folder
├── Dockerfile --service docker
├── firebase_config.py --service firebase
├── inference_config.py --service pretrained model
├── middleware.py --service middleware
└── requiremens.txt --list pustaka
