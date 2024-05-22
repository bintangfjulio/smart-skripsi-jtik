from models.lecturer import Lecturer
from flask import Blueprint, request, redirect, url_for
from middleware import role_required
from firebase_config import storage_upload_file


lecturer = Blueprint('lecturer', __name__, template_folder='templates', url_prefix='/dashboard/lecturer')


@lecturer.route('/create', methods=['POST'])
@role_required('admin')
def create():
    nama = request.form['nama']
    kompetensi = request.form['kompetensi']
    foto = storage_upload_file(request.files['foto'], 'lecturer')

    try:
        lecturer = Lecturer(nama, kompetensi, foto)
        lecturer.save()

        # flash('Tag berhasil dibuat', 'success')
        return redirect(url_for('dashboard.lecturer'))

    except Exception as e:
        print('[ERROR] [CREATE TAG]: ', e)
        # flash('Terjadi kesalahan server saat membuat tag', 'error')
        # return redirect(url_for('dashboard.tag'))
