from flask_login import UserMixin
from firebase_config import firebase_db
from firebase_admin import firestore


class User(UserMixin):
    def __init__(self, id=None, nama=None, email=None, role=None, registered_at=None, inactive=None):
        self.id = id
        self.nama = nama
        self.email = email
        self.role = role
        self.registered_at = registered_at
        self.inactive = inactive


    def update(self):
        firebase_db.collection('users').document(self.id).update({
            'inactive': self.inactive,
        })

        return self
    

    @staticmethod
    def fetch():
        users = firebase_db.collection('users').where('role', '==', 'pengguna').order_by("registered_at", direction=firestore.Query.DESCENDING).stream()

        datas = []
        for user in users:
            data = user.to_dict()
            datas.append({
                'id': user.id,
                'nama': data['nama'],
                'email': data['email'],
                'role': data['role'],
                'inactive': data['inactive'],
                'status_badge': 'danger' if data['inactive'] == "1" else 'success',
                'registered_at': data['registered_at'].strftime("%A, %d-%m-%Y %H:%M:%S")
            })

        return datas