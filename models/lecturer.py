from firebase_config import firebase_db


class Lecturer:
    def __init__(self, nama, kompetensi, foto, id=None):
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
    
    @staticmethod
    def fetch():
        lectures = firebase_db.collection('lecturer').order_by('nama').stream()

        datas = []
        for lecture in lectures:
            data = lecture.to_dict()
            datas.append({
                'id': lecture.id,
                'nama': data['nama'],
                'kompetensi': data['kompetensi'],
                'foto': data['foto']
            })

        return datas
    
    # def update(self):
    #     firebase_db.collection('lecturer').document(self.id).update({
    #         'nama': self.nama,
    #         'kompetensi': self.kompetensi,
    #         'foto': self.foto
    #     })

    #     return self
    
    # def delete(self):
    #     firebase_db.collection('lecturer').document(self.id).delete()

    #     return self
