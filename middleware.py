from flask_login import LoginManager, current_user
from models.user import User
from functools import wraps
from firebase_config import firebase_db
from flask import redirect, url_for


login_manager = LoginManager()

def init_middleware(app):
    login_manager.init_app(app)
    

@login_manager.user_loader
def load_user(id):    
    user_doc = firebase_db.collection('users').document(id).get()

    if user_doc.exists:
        user_data = user_doc.to_dict()
        return User(id=id, nama=user_data['nama'], email=user_data['email'], role=user_data['role'], registered_at=user_data['registered_at'], inactive=user_data['inactive'])
    
    return None


def role_required(*required_roles):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not current_user.is_authenticated or current_user.role not in required_roles:
                return redirect(url_for('unauthorized'))
            return func(*args, **kwargs)
        return wrapper
    
    return decorator