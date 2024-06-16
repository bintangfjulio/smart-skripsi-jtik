import io


from flask import redirect, Blueprint, request, url_for, flash, send_file
from docx import Document

file = Blueprint('file', __name__, template_folder='templates', url_prefix='/dashboard/file')

@file.route('/ekspor', methods=['POST'])
def ekspor():
    nama = request.form['nama']
    nim = request.form['nim']
    file_type = request.form['file_type']

    data = request.form.to_dict()

    doc = Document(f'resources/{file_type}.docx')

    try:
        for p in doc.paragraphs:
            for key, value in data.items():
                if f'{{{{{key}}}}}' in p.text:
                    p.text = p.text.replace(f'{{{{{key}}}}}', value)

        buffer = io.BytesIO()
        doc.save(buffer)
        buffer.seek(0)

        return send_file(buffer, as_attachment=True, download_name=f'{file_type}_{nama}_{nim}.docx', mimetype='application/vnd.openxmlformats-officedocument.wordprocessingml.document')

    except Exception as e:
        flash(('Ekspor Berkas Gagal', 'Berkas gagal diekspor'), 'error')