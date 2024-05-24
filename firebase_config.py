import os
import pyrebase
import tempfile
import uuid

from dotenv import load_dotenv
from firebase_admin import credentials, initialize_app, firestore, storage


load_dotenv(override=True)

firebase_config = {
    'apiKey': os.getenv('FIREBASE_API_KEY'),
    'authDomain': os.getenv('FIREBASE_AUTH_DOMAIN'),
    'projectId': os.getenv('FIREBASE_PROJECT_ID'),
    'storageBucket': os.getenv('FIREBASE_STORAGE_BUCKET'),
    'messagingSenderId': os.getenv('FIREBASE_MESSAGING_SENDER_ID'),
    'appId': os.getenv('FIREBASE_APP_ID'),
    'databaseURL': ''
}

firebase_admin_config = {
    'type': os.getenv('FIREBASE_ADMIN_TYPE'),
    'project_id': os.getenv('FIREBASE_ADMIN_PROJECT_ID'),
    'private_key_id': os.getenv('FIREBASE_ADMIN_PRIVATE_KEY_ID'),
    'private_key': os.getenv('FIREBASE_ADMIN_PRIVATE_KEY').replace('\\n', '\n'),
    'client_email': os.getenv('FIREBASE_ADMIN_CLIENT_EMAIL'),
    'client_id': os.getenv('FIREBASE_ADMIN_CLIENT_ID'),
    'auth_uri': os.getenv('FIREBASE_ADMIN_AUTH_URI'),
    'token_uri': os.getenv('FIREBASE_ADMIN_TOKEN_URI'),
    'auth_provider_x509_cert_url': os.getenv('FIREBASE_ADMIN_AUTH_PROVIDER_X509_CERT_URL'),
    'client_x509_cert_url': os.getenv('FIREBASE_ADMIN_CLIENT_X509_CERT_URL'),
    'universe_domain': os.getenv('FIREBASE_ADMIN_UNIVERSE_DOMAIN')
}

firebase = pyrebase.initialize_app(firebase_config)
firebase_admin = initialize_app(credentials.Certificate(firebase_admin_config), {
    'storageBucket': os.getenv('FIREBASE_STORAGE_BUCKET')
})

firebase_auth = firebase.auth()
firebase_db = firestore.client()
firebase_storage = storage.bucket()


def storage_upload_file(file, destination_folder):
    temp = tempfile.NamedTemporaryFile(delete=False)
    file.save(temp.name)
    temp.close()

    unique_filename = str(uuid.uuid4()) + '.' + file.filename.split('.')[-1]

    blob = firebase_storage.blob(f'{destination_folder}/{unique_filename}')
    blob.upload_from_filename(temp.name)
    blob.make_public()
    os.remove(temp.name)

    return blob.public_url


def storage_delete_file(public_url):
    firebase_storage.blob(public_url.replace(f'https://storage.googleapis.com/{os.getenv('FIREBASE_STORAGE_BUCKET')}/', '')).delete()