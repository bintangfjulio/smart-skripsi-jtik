from firebase_config import firebase_db


class Lecturer:
    def __init__(self, id=None, nama=None, kompetensi=None, foto=None):
        self.id = id
        self.nama = nama
        self.kompetensi = kompetensi
        self.foto = foto


    def save(self):
        _, doc = firebase_db.collection('lecturer').add({
            'nama': self.nama,
            'kompetensi': self.kompetensi,
            'foto': self.foto
        })

        self.id  = doc.id

        return self
    

    def delete(self):
        firebase_db.collection('lecturer').document(self.id).delete()

        return self
    
    
    @staticmethod
    def fetch():
        lectures = firebase_db.collection('lecturer').order_by('nama').stream()

        datas = []
        for lecturer in lectures:
            data = lecturer.to_dict()
            datas.append({
                'id': lecturer.id,
                'nama': data['nama'],
                'kompetensi': data['kompetensi'],
                'foto': data['foto']
            })

        return datas
