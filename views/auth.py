from flask import Blueprint, render_template, request, flash
from requests.exceptions import HTTPError

auth = Blueprint('auth', __name__, template_folder='templates')

@auth.route('/sign-in', methods=['GET', 'POST'])
def sign_in():
    return render_template('auth/sign_in.html')

@auth.route('/sign-up', methods=['GET', 'POST'])
def sign_up():
    if request.method == 'POST':
        nama = request.form['nama']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        if not nama or not email or not password or not confirm_password:
            return
        
        if password != confirm_password:
            return
        
    return render_template('auth/sign_up.html')