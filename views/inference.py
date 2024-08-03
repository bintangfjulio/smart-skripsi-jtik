from flask import request, Blueprint, jsonify, current_app
from middleware import role_required
from models.lecturer import Lecturer
from models.history import History
from datetime import datetime
from flask_login import current_user
from pytz import timezone


inference = Blueprint('inference', __name__, template_folder='templates', url_prefix='/dashboard/inference')
tz = timezone('Asia/Jakarta')

@inference.route('/inferencer', methods=['POST'])
@role_required('pengguna')
def inferencer():
    abstrak = request.get_json().get('abstrak')
    kata_kunci = request.get_json().get('kata_kunci')
    klasifikasi = request.get_json().get('klasifikasi')
    rekomendasi = request.get_json().get('rekomendasi')

    try:
        text = current_app.inference.text_processing(abstrak, kata_kunci)

        if klasifikasi:
            probs, kbk = current_app.inference.classification(text)
            lecturers = Lecturer.fetch(kelompok_bidang_keahlian=kbk) 

        else:
            probs = ""
            kbk = ""
            lecturers = ""

        if rekomendasi:
            recommended = current_app.inference.content_based_filtering(text)
            
        else:
            recommended = ""

        history = History(abstrak=abstrak, kata_kunci=kata_kunci, probabilitas=probs, kelompok_bidang_keahlian=kbk, tanggal_inferensi=datetime.now(tz), top_similarity=recommended)
        history.save(current_user.id)

        return jsonify(message={'probs': probs, 'lecturers': lecturers, 'kbk': kbk, 'top_similarity': recommended}, status="success"), 200
    
    except Exception as e:
        return jsonify(message={'error': 'Server error'}, status="error"), 500