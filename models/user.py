from flask_login import UserMixin
from firebase_config import firebase_db


class User(UserMixin):
    def __init__(self, id=None, nama=None, email=None, role=None, registered_at=None, inactive=None):
        self.id = id
        self.nama = nama
        self.email = email
        self.role = role
        self.registered_at = registered_at
        self.inactive = inactive

    @staticmethod
    def fetch():
        users = firebase_db.collection('users').where('role', '==', 'pengguna').order_by('nama').stream()

        datas = []
        for user in users:
            data = user.to_dict()
            datas.append({
                'id': user.id,
                'nama': data['nama'],
                'email': data['email'],
                'role': data['role'],
                'inactive': data['inactive'],
                'registered_at': data['registered_at']
            })

        return datas