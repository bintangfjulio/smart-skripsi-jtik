from firebase_config import firebase_db


class Lecturer:
    def __init__(self, id=None, nama=None, kelompok_bidang_keahlian=None, foto=None):
        self.id = id
        self.nama = nama
        self.kelompok_bidang_keahlian = kelompok_bidang_keahlian
        self.foto = foto


    def save(self):
        _, doc = firebase_db.collection('lecturers').add({
            'nama': self.nama,
            'kelompok_bidang_keahlian': self.kelompok_bidang_keahlian,
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
            'kelompok_bidang_keahlian': self.kelompok_bidang_keahlian,
            'foto': self.foto
        })

        return self
    
    
    @staticmethod
    def fetch(kelompok_bidang_keahlian=None):
        query = firebase_db.collection('lecturers').order_by('nama')

        if kelompok_bidang_keahlian is not None:
            query = query.where('kelompok_bidang_keahlian', '==', kelompok_bidang_keahlian)

        lectures = query.stream()

        datas = []
        for lecturer in lectures:
            data = lecturer.to_dict()
            datas.append({
                'id': lecturer.id,
                'nama': data['nama'],
                'kelompok_bidang_keahlian': data['kelompok_bidang_keahlian'],
                'foto': data['foto']
            })

        return datas
