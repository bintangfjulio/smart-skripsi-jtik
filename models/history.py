import arrow

from firebase_config import firebase_db
from firebase_admin import firestore


class History:
    def __init__(self, id=None, abstrak=None, kata_kunci=None, probabilitas=None, kelompok_bidang_keahlian=None, tanggal_inferensi=None, top_similarity=None):
        self.id = id
        self.abstrak = abstrak
        self.kata_kunci = kata_kunci
        self.probabilitas = probabilitas
        self.kelompok_bidang_keahlian = kelompok_bidang_keahlian
        self.tanggal_inferensi = tanggal_inferensi  
        self.top_similarity = top_similarity


    def save(self, id):
        _, doc = firebase_db.collection('users').document(id).collection('histories').add({
            'abstrak': self.abstrak,
            'kata_kunci': self.kata_kunci,
            'probabilitas': self.probabilitas,
            'kelompok_bidang_keahlian': self.kelompok_bidang_keahlian,
            'tanggal_inferensi': self.tanggal_inferensi,
            'top_similarity': self.top_similarity
        })
        
        self.id = doc.id

        return self
    

    def delete(self, id):
        firebase_db.collection('users').document(id).collection('histories').document(self.id).delete()

        return self
    

    @staticmethod
    def fetch(id):
        histories = firebase_db.collection('users').document(id).collection('histories').order_by("tanggal_inferensi", direction=firestore.Query.DESCENDING).stream()

        datas = []
        for history in histories:
            data = history.to_dict()
            datas.append({
                'id': history.id,
                'abstrak': data['abstrak'],
                'kata_kunci': data['kata_kunci'],
                'probabilitas': dict(sorted(data['probabilitas'].items(), key=lambda item: item[1], reverse=True)),
                'kelompok_bidang_keahlian': data['kelompok_bidang_keahlian'],
                'tanggal_inferensi': arrow.get(data['tanggal_inferensi']).to('Asia/Jakarta').format('dddd, DD-MM-YYYY HH:mm:ss', locale='id_ID'),
                'top_similarity': data['top_similarity']
            })

        return datas