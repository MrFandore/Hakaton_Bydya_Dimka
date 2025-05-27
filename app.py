from flask import Flask, render_template, request, redirect, url_for, flash, abort
from flask_login import LoginManager, UserMixin, login_user, login_required, current_user, logout_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'

# Инициализация Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'index'

# Папка для загруженных файлов
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static/uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALLOWED_EXTENSIONS = {'mp3', 'wav'}

# Временные данные (заглушка для БД)
users_db = {
    'admin': {'password': generate_password_hash('admin123'), 'role': 'admin'}
}
audio_files = []
processing_files = []

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

class User(UserMixin):
    def __init__(self, id):
        self.id = id
        self.role = users_db.get(id, {}).get('role', 'user')

@login_manager.user_loader
def load_user(user_id):
    return User(user_id) if user_id in users_db else None

@app.route('/', methods=['GET', 'POST'])
def index():
    if current_user.is_authenticated:
        return redirect(url_for('player'))

    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        action = request.form.get('action')

        if action == 'login':
            if username in users_db and check_password_hash(users_db[username]['password'], password):
                user = User(username)
                login_user(user)
                return redirect(url_for('player'))
            flash('Неверные данные', 'error')
        elif action == 'register':
            if username in users_db:
                flash('Имя занято', 'error')
            else:
                users_db[username] = {'password': generate_password_hash(password), 'role': 'user'}
                login_user(User(username))
                flash('Регистрация успешна!', 'success')
                return redirect(url_for('player'))

    return render_template('index.html')

@app.route('/player')
@login_required
def player():
    return render_template('player.html', files=audio_files)

@app.route('/file/<int:file_id>')
@login_required
def file_detail(file_id):
    file = next((f for f in audio_files if f['id'] == file_id), None)
    if not file:
        abort(404)
    return render_template('file_detail.html', file=file)

@app.route('/admin')
@login_required
def admin_dashboard():
    if current_user.role != 'admin':
        abort(403)
    return render_template('admin/dashboard.html', files=audio_files)

# Загрузка нового файла
@app.route('/admin/upload', methods=['GET', 'POST'])
@login_required
def admin_upload():
    if current_user.role != 'admin':
        abort(403)
    if request.method == 'POST':
        file = request.files.get('file')
        title = request.form['title']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)

            # Добавляем в обработку (эмуляция)
            new_id = len(audio_files) + len(processing_files) + 1
            processing_files.append({
                'id': new_id,
                'title': title,
                'filename_original': filename,
                'filename_processed': filename,  # Заглушка
                'tags': '',
                'author': '',
                'year': 2025,
                'text': 'Распознанный текст (заглушка)'
            })
            flash('Файл загружен и отправлен на обработку!', 'success')
            return redirect(url_for('admin_dashboard'))
        flash('Ошибка загрузки файла!', 'error')
    return render_template('admin/upload.html')

# Проверка и редактирование обработанных файлов
@app.route('/admin/processing', methods=['GET', 'POST'])
@login_required
def admin_processing():
    if current_user.role != 'admin':
        abort(403)
    if request.method == 'POST':
        file_id = int(request.form['file_id'])
        file = next((f for f in processing_files if f['id'] == file_id), None)
        if file:
            file['tags'] = request.form['tags']
            file['text'] = request.form['text']
            # Подтверждаем - переносим в общий список
            audio_files.append(file)
            processing_files.remove(file)
            flash('Аудио успешно подтверждено!', 'success')
        return redirect(url_for('admin_processing'))
    return render_template('admin/processing.html', files=processing_files)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
