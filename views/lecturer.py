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

    try:
        foto = storage_upload_file(request.files['foto'], 'lecturer')
        lecturer = Lecturer(nama=nama, kompetensi=kompetensi, foto=foto)
        lecturer.save()
        
        flash(('Tambah Data Sukses', 'Data dosen berhasil ditambahkan'), 'success')

    except Exception as e:
        flash(('Tambah Data Gagal', 'Terjadi kesalahan server saat menambahkan'), 'error')
        
    return redirect(url_for('dashboard.lecturer'))


@lecturer.route('/update', methods=['POST'])
@role_required('admin')
def update():
    id = request.form['id']
    nama = request.form['nama']
    kompetensi = request.form['kompetensi']
    foto = request.form['prev_foto']
    
    if request.files['foto'].filename != '':
        storage_delete_file(foto)
        foto = storage_upload_file(request.files['foto'], 'lecturer')

    try:
        lecturer = Lecturer(id=id, nama=nama, kompetensi=kompetensi, foto=foto)
        lecturer.update()

        flash(('Perbarui Data Sukses', 'Data dosen berhasil diperbarui'), 'success')

    except Exception as e:
        flash(('Perbarui Data Gagal', 'Terjadi kesalahan server saat memperbarui'), 'error')
        
    return redirect(url_for('dashboard.lecturer'))

@lecturer.route('/delete', methods=['POST'])
@role_required('admin')
def delete():
    id = request.form['id']
    foto = request.form['foto']

    try:
        lecturer = Lecturer(id=id, foto=foto)
        lecturer.delete()
        storage_delete_file(lecturer.foto)

        flash(('Hapus Data Sukses', 'Data dosen berhasil dihapus'), 'success')

    except Exception as e:
        flash(('Hapus Data Gagal', 'Terjadi kesalahan server saat menghapus'), 'error')
        
    return redirect(url_for('dashboard.lecturer'))
