import jsonify

from flask import Blueprint, request, redirect, url_for, flash
from middleware import role_required


classifier = Blueprint('classifier', __name__, template_folder='templates', url_prefix='/dashboard/classifier')

@classifier.route('/inference', methods=['POST'])
@role_required('pengguna')
def inference():
    id = request.form['id']
    keywords = request.form['keywords']
    abstrak = request.form['abstrak']

    if not keywords or not abstrak:
        flash(('Klasifikasi Gagal', 'Seluruh isian harus diisi'), 'error')
    
    try:
        text = keywords + " - " + abstrak

    except Exception as e:
        flash(('Klasifikasi Gagal', 'Terjadi kesalahan server saat klasifikasi'), 'error')
        
    return redirect(url_for('dashboard.classifier'))
