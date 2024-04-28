import argparse
import emoji
import re
import torch
import pandas as pd

from torch import clamp
from transformers import BertTokenizer, BertModel
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.metrics.pairwise import cosine_similarity
from model.bert_cnn import BERT_CNN


def preprocessor(text):
    text = text.lower()
    text = emoji.replace_emoji(text, replace='') 
    text = re.sub(r'\n', ' ', text) 
    text = re.sub(r'http\S+', '', text)  
    text = re.sub(r'\d+', '', text)  
    text = re.sub(r'[^a-zA-Z ]', '', text)  
    text = ' '.join([word for word in text.split() if word not in stop_words])  
    text = stemmer.stem(text)
    text = text.strip()   
    
    return text

def classification(text):
    token = tokenizer.encode_plus(
        text=text,
        add_special_tokens=True,
        max_length=config["max_length"],
        return_tensors='pt',
        padding="max_length", 
        truncation=True)
    
    pretrained_bert = BertModel.from_pretrained(config["bert_model"], output_attentions=False, output_hidden_states=True)
    model = BERT_CNN(pretrained_bert=pretrained_bert, dropout=config["dropout"], window_sizes=config["window_sizes"], in_channels=config["in_channels"], out_channels=config["out_channels"], num_bert_states=config["num_bert_states"])
    model.load_state_dict(torch.load('checkpoints/flat_model.pt', map_location=device))
    model.to(device)

    model.eval()
    with torch.no_grad():
        preds = model(input_ids=token["input_ids"].to(device), attention_mask=token["attention_mask"].to(device))
        result = torch.argmax(preds, dim=1)

    return labels[result[0]]

def similarity_checker(text_1, text_2):
    token = tokenizer([text_1, text_2], max_length=config['max_length'], padding="max_length", truncation=True,  return_tensors='pt') 
    model = BertModel.from_pretrained(config["bert_model"], output_attentions=False, output_hidden_states=False)

    attention_mask = token.attention_mask

    outputs = model(**token)
    embeddings = outputs.last_hidden_state

    mask = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
    masked_embeddings = embeddings * mask

    summed = masked_embeddings.sum(1)
    counts = clamp(mask.sum(1), min=1e-9)
    mean_pooled = (summed / counts).detach().numpy()

    similary = cosine_similarity([mean_pooled[0]], [mean_pooled[1]])   
    
    return similary[0][0]

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='data_repo_jtik.csv')
    parser.add_argument("--bert_model", type=str, default="indolem/indobert-base-uncased")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max_length", type=int, default=360)
    parser.add_argument("--in_channels", type=int, default=4)
    parser.add_argument("--out_channels", type=int, default=32)
    parser.add_argument("--window_sizes", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--num_bert_states", type=int, default=4)
    config = vars(parser.parse_args())

    dataset = pd.read_csv(f'dataset/{config["dataset"]}')
    stop_words = StopWordRemoverFactory().get_stop_words()
    tokenizer = BertTokenizer.from_pretrained(config["bert_model"], use_fast=False)
    stemmer = StemmerFactory().create_stemmer()
    labels = sorted(dataset['prodi'].unique().tolist())

    responses = [
        {"dosen": "A", "abstrak": '"analisis perbandingan performa open source hypervisor proxmox ve, xen hypervisor, kvm, dan xcp-ng","sebuah komputer server yang akan dilakukan instalasi empat hypervisor proxmox virtual environment, xen hypervisor, kvm (berbasis ubuntu server 18.04), dan xcp-ng secara bergantian kemudian diinstal satu sistem operasi guest ubuntu server 18.04 pada masing-masing hypervisor. setelah instalasi hypervisor dan sistem operasi guest selesai lalu dilakukan pengujian pada sistem operasi guest dengan parameter meliputi cpu throughput, cpu processing speed, memory bandwidth, kecepatan baca-tulis disk, dan disk latency."'},
        {"dosen": "A", "abstrak": 'rancang bangun aplikasi manajemen aset bergerak dengan geolocation dan haversine berbasis mobile,"perusahaan pt. sucofindo advisory utama dalam satu bulan mengeluarkan aset sebanyak tujuh unit mobil dan lima unit ponsel untuk melakukan operasional perusahaan yang berupa survei. namun aset bergerak tersebut memiliki kendala yaitu pada penyalahgunaan pada peminjaman aset yang mengkibatkan perusahaan mengalami kekurangan jumlah aset akibat keterlambatan pengembalian. dalam periode bulan mei hingga desember 2020 didapatkan presentase angka penyalahgunaan aset sebesar 24%. permasalahan ini disebabkan karena perusahaan tidak dapat melakukan pelacakan aset bergerak. dampak kerugian yang timbul dari penyalahgunaan aset tersebut adalah tidak mencukupinya ketersediaan aset bergerak yang akan digunakan untuk operasional perusahaan sehingga perusahaan diharuskan menyewa. rancang bangun sistem akan dibangun pada dua platform yakni berbasis mobile menggunakan framework flutter dan web menggunakan framework laravel. aplikasi mobile digunakan sebagai alat pelacak oleh peminjam aset akan aktif secara kontinu untuk pengiriman lokasi dengan geolocation. formula haversine digunakan untuk mendeteksi aset sesuai dengan lokasi tempat penugasan, apabila tidak digunakan pada lokasi tempat penugasan maka status peminjaman aset akan diubah menjadi penyalahgunaan aset sedangkan aplikasi web digunakan sebagai pemantauan aset, manajemen data penugasan, manajemen data aset, dan manajemen data lokasi. sistem ini dapat menentukan status penyalahgunaan aset dan sistem dapat digunakan untuk mengelola aset bergerak dari yang sebelumnya menggunakan excel serta dapat memberikan informasi laporan peminjaman aset bergerak perusahaan."'},
        {"dosen": "B", "abstrak": 'analisis perbandingan sistem pendukung keputusan menggunakan metode waspas dan wp dalam seleksi peserta olimpiade sains di smpn 174 jakarta,"olimpiade sains nasional (osn) merupakan ajang kompetisi di bidang sains tahunan yang dilaksanakan oleh pemerintah dalam bidang pendidikan bagi para siswa dalam berbagai jenjang seperti sekolah menengah pertama. smpn 174 jakarta merupakan salah satu sekolah yang ikut serta bahkan pernah menjuarai olimpiade sains. untuk dapat mengikuti olimpiade, setiap sekolah hanya dapat mengirimkan perwakilan, sehingga dibutuhkan penyeleksian terhadap siswa. pada seleksi manual, ditemukan banyak kendala yang dihadapi karena banyaknya calon peserta dan juga adanya pemilihan yang bersifat subjektif. oleh karena itu, dibutuhkan sistem pendukung keputusan untuk mengatasi permasalahan tersebut. dalam membuat sistem ini, digunakan dua metode yaitu weighted agregated sum product assessment (waspas) dan weighted product (wp). kedua metode ini dibandingkan untuk mendapatkan metode yang lebih cocok untuk diterapkan di kasus seleksi peserta osn. kriteria yang digunakan yaitu nilai ipa, nilai m3atematika, nilai rata-rata rapor, presensi, dan nilai uji test olimpiade. sistem melakukan penilaian dan hasil perankingan di setiap metode. perbandingan kedua metode ini menghasilkan bahwa metode waspas memiliki jarak perbandingan nilai yang lebih jauh di setiap rankingnya dibandingkan metode wp, sehingga diperoleh metode yang lebih cocok yaitu metode waspas."'},
        {"dosen": "B", "abstrak": 'implementasi sistem keamanan dan kendali gerbang otomatis dengan menggunakan sensor pir dan sensor getaran berbasis arduino uno,"pada umumnya penggunaan pintu gerbang dilakukan secara manual. hal ini menimbulkan kurangnya tingkat efisiensi terutama pada pemilik rumah. untuk itu, diusulkan sistem kendali buka tutup gerbang dengan menggunakan aplikasi telegram dan dilengkapi dengan sistem keamanan yang akan memberikan notifikasi ke pemilik dalam implementasinya. metode yang digunakan pada penelitian ini adalah metode deskriptif dengan pendekatan kuantitatif berdasarkan implementasi sistem yang dibuat. sistem ini menggunakan aplikasi telegram sebagai kontrol buka tutup gerbang dan notifikasi sensor pir dan sensor sw-420. alat ini menggunakan arduino dan esp 32 cam sebagai media pengiriman dan penerimaan data, cara kerja alat ini menggunakan perintah bot telegram yang akan di proses oleh esp32 cam, didalam bot telegram terdapat menu buka dan tutup gerbang,mengaktifkan dan mematikan sensor pir , mengaktifkan dan mematikan sensor sw-420. ketika bot telegram memberikan perintah /piron maka sensor pir akan menyala dan di saat terjadi gerkan disekitar sensor maka esp32 cam akan mengirimkan foto ke telegram. ketika bot telegram mengirimkan perintah /getaron maka sensor getaran akan menyala jika terjadi getaran akan mengirimkan sinyal notifikasi dan buzzer berbunyi, ketika bot telegram memebrikan perintah /buka atau /tutup maka otomatis pintu akan terbuka maupun tertutup sesuai dengan perintah yang di terima"'},
        {"dosen": "C", "abstrak": 'pengembangan sistem evaluasi pelatihan menggunakan model kirkpatrick,"sistem evaluasi pelatihan menggunakan model kirkpatrick adalah aplikasi yang dirancang untuk membantu organisasi mengevaluasi efektivitas pelatihan mereka berdasarkan model kirkpatrick, yang terkenal dengan empat tingkatan evaluasi."'}
    ]
    
    abstract = input("Please input abstract: ")
    abstract = preprocessor(abstract)
    
    print("Classification...")
    classified = classification(abstract)
    print(f"\nClassification Result: {classified}")

    print("Similarity Check...")
    highest_dosen_similarity = {}
    for item in responses:
        abstrak_response = preprocessor(item["abstrak"])
        similiraty_score = similarity_checker(abstract, abstrak_response)

        if item["dosen"] in highest_dosen_similarity:
            if similiraty_score > highest_dosen_similarity[item["dosen"]]:
                highest_dosen_similarity[item["dosen"]] = similiraty_score
        else:
            highest_dosen_similarity[item["dosen"]] = similiraty_score

    sorted_scores = sorted(highest_dosen_similarity.items(), key=lambda x: x[1], reverse=True)
    for i, (dosen, score) in enumerate(sorted_scores):
        print(f"{i+1}. Dosen: {dosen}, Similarity Score: {score}")