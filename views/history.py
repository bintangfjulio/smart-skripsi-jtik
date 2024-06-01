from flask import Blueprint, request, redirect, url_for, flash
from middleware import role_required
from models.history import History
from flask_login import current_user


history = Blueprint('history', __name__, template_folder='templates', url_prefix='/dashboard/history')

@history.route('/delete', methods=['POST'])
@role_required('pengguna')
def delete():
    id = request.form['id']

    try:
        history = History(id=id)
        history.delete(current_user.id)

        flash(('Hapus Riwayat Sukses', 'Riwayat berhasil dihapus'), 'success')

    except Exception as e:
        flash(('Hapus Riwayat Gagal', 'Terjadi kesalahan server saat menghapus'), 'error')
        
    return redirect(url_for('dashboard.history', id=current_user.id))
