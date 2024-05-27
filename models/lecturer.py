from firebase_config import firebase_db


class Lecturer:
    def __init__(self, id=None, nama=None, kompetensi=None, foto=None):
        self.id = id
        self.nama = nama
        self.kompetensi = kompetensi
        self.foto = foto


    def save(self):
        _, doc = firebase_db.collection('lecturers').add({
            'nama': self.nama,
            'kompetensi': self.kompetensi,
            'foto': self.foto
        })

        self.id  = doc.id

        return self
    

    def delete(self):
        firebase_db.collection('lecturers').document(self.id).delete()

        return self
    

    def update(self):
        firebase_db.collection('lecturers').document(self.id).update({
            'nama': self.nama,
            'kompetensi': self.kompetensi,
            'foto': self.foto
        })

        return self
    
    
    @staticmethod
    def fetch(kompetensi=None):
        query = firebase_db.collection('lecturers').order_by('nama')

        if kompetensi is not None:
            query = query.where('kompetensi', '==', kompetensi)

        lectures = query.stream()

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
