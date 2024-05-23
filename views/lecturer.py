from models.lecturer import Lecturer
from flask import Blueprint, request, redirect, url_for, flash
from middleware import role_required
from firebase_config import storage_upload_file, storage_delete_file


lecturer = Blueprint('lecturer', __name__, template_folder='templates', url_prefix='/dashboard/lecturer')

@lecturer.route('/create', methods=['POST'])
@role_required('admin')
def create():
    nama = request.form['nama']
    kompetensi = request.form['kompetensi']
    foto = storage_upload_file(request.files['foto'], 'lecturer')

    try:
        lecturer = Lecturer(nama=nama, kompetensi=kompetensi, foto=foto)
        lecturer.save()
        flash(('Tambah Data Sukses', 'Data dosen berhasil ditambahkan'), 'success')

    except Exception as e:
        flash(('Tambah Data Gagal', 'Terjadi kesalahan server saat menambahkan'), 'error')
        
    return redirect(url_for('dashboard.lecturer'))


@lecturer.route('/delete', methods=['POST'])
@role_required('admin')
def delete():
    id = request.form['id']

    try:
        lecturer = Lecturer(id=id)
        lecturer.delete()
        storage_delete_file(lecturer.foto)

    except Exception as e:
        print(e)
        
    return redirect(url_for('dashboard.lecturer'))
