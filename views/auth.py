import json

from flask import Blueprint, render_template, request, flash, redirect, url_for
from requests.exceptions import HTTPError
from firebase_config import firebase_auth, firebase_db

from flask_login import logout_user, login_user
from middleware import load_user


auth = Blueprint('auth', __name__, template_folder='templates')

@auth.route('/sign-up', methods=['GET', 'POST'])
def sign_up():
    if request.method == 'POST':
        nama = request.form['nama']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        if not nama or not email or not password or not confirm_password:
            flash(('Pendaftaran Gagal', 'Semua isian harus diisi'), 'error')
            return render_template('auth/sign_up.html')
        
        if password != confirm_password:
            flash(('Pendaftaran Gagal', 'Password dan konfirmasi password tidak sama'), 'error')
            return render_template('auth/sign_up.html')
        
        try:
            user = firebase_auth.create_user_with_email_and_password(email, password)

            data = {
                'nama': nama,
                'email': email,
                'role': 'user'
            }

            firebase_db.collection('users').document(user['localId']).set(data)

            firebase_auth.send_email_verification(user['idToken'])
            flash(('Pendaftaran Diproses', 'Periksa email untuk verifikasi akun'), 'success')

        except HTTPError as e:
            message = json.loads(e.strerror)['error']['message']

            if 'WEAK_PASSWORD' in message:
                message = 'Password minimal 6 karakter'

            if 'EMAIL_EXISTS' in message:
                message = 'Email sudah terdaftar'

            flash(('Pendaftaran Gagal', message), 'error')

    return render_template('auth/sign_up.html')


@auth.route('/sign-in', methods=['GET', 'POST'])
def sign_in():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        if not email or not password:
            flash(('Masuk Gagal', 'Semua isian harus diisi'), 'error')
            return render_template('auth/sign_in.html')
        
        try:
            user = firebase_auth.sign_in_with_email_and_password(email, password)
            user_info = firebase_auth.get_account_info(user['idToken'])['users'][0]

            if user_info['emailVerified'] == False:
                flash(('Masuk Gagal', 'Email belum diverifikasi'), 'error')
                return render_template('auth/sign_in.html')

        except HTTPError as e:
            message = json.loads(e.strerror)['error']['message']

            if 'INVALID_LOGIN_CREDENTIALS' in message:
                message = 'Email atau password salah'

            flash(('Masuk Gagal', message), 'error')
            return render_template('auth/sign_in.html')
        
        except Exception as e:
            flash(('Masuk Gagal', 'Server sedang bermasalah'), 'error')
            return render_template('auth/sign_in.html')
        
        user_id = user_info['localId']
        user = load_user(user_id)

        if user:
            login_user(user)
            return redirect(url_for('dashboard.lecturer'))

    return render_template('auth/sign_in.html')


@auth.route('/reset-password', methods=['GET', 'POST'])
def reset_password():
    if request.method == 'POST':
        email = request.form['email']

        if not email:
            flash(('Reset Password Gagal', 'Email harus diisi'), 'error')
            return render_template('auth/reset_password.html')
        
        try: 
            firebase_auth.send_password_reset_email(email)
            flash(('Reset Password Diproses', 'Periksa email untuk reset password'), 'success')

        except HTTPError as e:
            message = json.loads(e.strerror)['error']['message']

            if 'INVALID_EMAIL' in message:
                message = 'Email tidak ditemukan' 

            flash(('Reset Password Gagal', message), 'error')
        
        except Exception as e:
            flash(('Reset Password Gagal', 'Server sedang bermasalah'), 'error')

    return render_template('auth/reset_password.html')


@auth.route('/sign-out')
def sign_out():
    logout_user()
    return redirect(url_for('auth.sign_in'))