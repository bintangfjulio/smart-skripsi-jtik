from flask import Blueprint, request, redirect, url_for, flash
from middleware import role_required
from models.user import User


user = Blueprint('user', __name__, template_folder='templates', url_prefix='/dashboard/user')

@user.route('/update', methods=['POST'])
@role_required('admin')
def update():
    id = request.form['id']
    status = request.form['status']

    try:
        user = User(id=id, inactive=status)
        user.update()

        flash(('Perbarui Status Sukses', 'Status pengguna berhasil diperbarui'), 'success')

    except Exception as e:
        flash(('Perbarui Status Gagal', 'Terjadi kesalahan server saat memperbarui'), 'error')
        
    return redirect(url_for('dashboard.user'))
