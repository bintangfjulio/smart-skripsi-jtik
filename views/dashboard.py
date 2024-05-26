from flask import Blueprint, render_template
from middleware import role_required
from models.lecturer import Lecturer
from models.user import User


dashboard = Blueprint('dashboard', __name__, template_folder='templates', url_prefix='/dashboard')

@dashboard.route('/lecturer', methods=['GET'])
@role_required('admin')
def lecturer():
    lecturers = Lecturer.fetch()
    return render_template('dashboard/lecturer.html', page="lecturer", lecturers=lecturers)


@dashboard.route('/user', methods=['GET'])
@role_required('admin')
def user():
    users = User.fetch()
    return render_template('dashboard/user.html', page="user", users=users)


@dashboard.route('/classifier', methods=['GET'])
@role_required('pengguna')
def classifier():
    return render_template('dashboard/classifier.html', page="classifier")