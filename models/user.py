from flask_login import UserMixin


class User(UserMixin):
    def __init__(self, id=None, nama=None, email=None, role=None):
        self.id = id
        self.nama = nama
        self.email = email
        self.role = role