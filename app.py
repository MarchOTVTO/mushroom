from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
import jwt
import datetime
import os
import sys
import traceback
from functools import wraps
import numpy as np
import io
from PIL import Image
from tensorflow.keras.models import load_model
import webbrowser
from threading import Timer

# 强制实时输出（解决打包后缓冲问题）
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None
sys.stderr.reconfigure(line_buffering=True) if hasattr(sys.stderr, 'reconfigure') else None

def log(msg):
    print(msg, flush=True)

# -------------------------- 初始化 Flask --------------------------
app = Flask(__name__)
CORS(app)

# -------------------------- 配置 --------------------------
basedir = os.path.abspath(os.path.dirname(__file__))
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir, 'instance', 'mushroom.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = 'your-secret-key-change-in-production'
app.config['JWT_EXPIRATION_HOURS'] = 168

db = SQLAlchemy(app)
bcrypt = Bcrypt(app)

os.makedirs(os.path.join(basedir, 'instance'), exist_ok=True)


# -------------------------- 数据库模型 --------------------------
class User(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)

    def set_password(self, password):
        self.password_hash = bcrypt.generate_password_hash(password).decode('utf-8')

    def check_password(self, password):
        return bcrypt.check_password_hash(self.password_hash, password)

    def to_dict(self):
        return {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'created_at': self.created_at.isoformat()
        }


class History(db.Model):
    __tablename__ = 'history'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    mushroom_name = db.Column(db.String(100), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    image_url = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)

    user = db.relationship('User', backref=db.backref('histories', lazy=True))

    def to_dict(self):
        return {
            'id': self.id,
            'mushroom_name': self.mushroom_name,
            'confidence': self.confidence,
            'image_url': self.image_url,
            'created_at': self.created_at.isoformat()
        }


# -------------------------- JWT 辅助函数 --------------------------
def generate_token(user_id):
    payload = {
        'user_id': user_id,
        'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=app.config['JWT_EXPIRATION_HOURS'])
    }
    return jwt.encode(payload, app.config['SECRET_KEY'], algorithm='HS256')


def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        auth_header = request.headers.get('Authorization')
        if auth_header and auth_header.startswith('Bearer '):
            token = auth_header.split(' ')[1]

        if not token:
            return jsonify({'success': False, 'message': '缺少认证令牌'}), 401

        try:
            payload = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
            current_user = User.query.get(payload['user_id'])
            if not current_user:
                return jsonify({'success': False, 'message': '用户不存在'}), 401
        except jwt.ExpiredSignatureError:
            return jsonify({'success': False, 'message': '令牌已过期'}), 401
        except jwt.InvalidTokenError:
            return jsonify({'success': False, 'message': '无效令牌'}), 401

        return f(current_user, *args, **kwargs)

    return decorated


# -------------------------- 认证路由 --------------------------
@app.route('/api/auth/register', methods=['POST'])
def register():
    data = request.get_json()
    username = data.get('username')
    email = data.get('email')
    password = data.get('password')

    if not username or not email or not password:
        return jsonify({'success': False, 'message': '用户名、邮箱和密码均为必填项'}), 400
    if len(password) < 6:
        return jsonify({'success': False, 'message': '密码长度至少为6位'}), 400

    if User.query.filter((User.username == username) | (User.email == email)).first():
        return jsonify({'success': False, 'message': '用户名或邮箱已被注册'}), 400

    user = User(username=username, email=email)
    user.set_password(password)
    db.session.add(user)
    db.session.commit()

    token = generate_token(user.id)
    return jsonify({
        'success': True,
        'message': '注册成功',
        'token': token,
        'user': user.to_dict()
    }), 201


@app.route('/api/auth/login', methods=['POST'])
def login():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')

    if not email or not password:
        return jsonify({'success': False, 'message': '邮箱和密码均为必填项'}), 400

    user = User.query.filter_by(email=email).first()
    if not user or not user.check_password(password):
        return jsonify({'success': False, 'message': '邮箱或密码错误'}), 401

    token = generate_token(user.id)
    return jsonify({
        'success': True,
        'message': '登录成功',
        'token': token,
        'user': user.to_dict()
    }), 200


@app.route('/api/auth/me', methods=['GET'])
@token_required
def get_current_user(current_user):
    return jsonify({'success': True, 'user': current_user.to_dict()}), 200


# -------------------------- 历史记录路由 --------------------------
@app.route('/api/history', methods=['POST'])
@token_required
def add_history(current_user):
    data = request.get_json()
    mushroom_name = data.get('mushroom_name')
    confidence = data.get('confidence')
    image_url = data.get('image_url')

    if not all([mushroom_name, confidence is not None, image_url]):
        return jsonify({'success': False, 'message': '缺少必要字段'}), 400

    history = History(
        user_id=current_user.id,
        mushroom_name=mushroom_name,
        confidence=confidence,
        image_url=image_url
    )
    db.session.add(history)
    db.session.commit()

    return jsonify({'success': True, 'message': '历史记录已保存', 'history': history.to_dict()}), 201


@app.route('/api/history', methods=['GET'])
@token_required
def get_history(current_user):
    histories = History.query.filter_by(user_id=current_user.id) \
        .order_by(History.created_at.desc()) \
        .limit(50) \
        .all()
    return jsonify({
        'success': True,
        'history': [h.to_dict() for h in histories]
    }), 200


@app.route('/api/history', methods=['DELETE'])
@token_required
def clear_history(current_user):
    History.query.filter_by(user_id=current_user.id).delete()
    db.session.commit()
    return jsonify({'success': True, 'message': '历史记录已清空'}), 200


# -------------------------- 模型预测 --------------------------
def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


MODEL_PATH = resource_path("mushroom_classifier_final.h5")
IMG_SIZE = (150, 150)

TRAIN_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data(1)', 'split_data', 'train')


def get_class_names():
    if os.path.exists(TRAIN_DATA_DIR):
        class_names = sorted(os.listdir(TRAIN_DATA_DIR))
        log(f"✅ 从训练目录加载了 {len(class_names)} 个类别：{class_names}")
        return class_names
    else:
        fallback = [
            "白毒伞", "毒蝇伞", "海鲜菇", "红菇", "猴头菇", "鸡枞菌", "鸡油菌",
            "金针菇", "口蘑", "灵芝", "牛肝菌", "平菇", "乳菇", "松茸", "香菇",
            "小松菇", "羊肚菌", "杏鲍菇", "竹荪"
        ]
        log(f"⚠️ 训练目录不存在，使用硬编码类别列表：{fallback}")
        return fallback


CLASS_NAMES = get_class_names()

# 加载模型（强制刷新输出）
log("🔍 正在加载模型...")
try:
    model = load_model(MODEL_PATH, compile=False)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    log(f"✅ 模型 {MODEL_PATH} 加载成功！")
    log(f"✅ 当前类别顺序（共{len(CLASS_NAMES)}类）：{CLASS_NAMES}")
except Exception as e:
    log(f"❌ 模型加载失败：{str(e)}")
    traceback.print_exc(file=sys.stdout)
    traceback.print_exc()  # 确保输出
    model = None


def preprocess_image(img, target_size):
    img = img.convert("RGB")
    img = img.resize(target_size, Image.Resampling.LANCZOS)
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({"success": False, "error": "模型未正常加载"}), 500

    if 'file' not in request.files:
        return jsonify({"success": False, "error": "未检测到上传的图片文件"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"success": False, "error": "未选择图片文件"}), 400

    try:
        img = Image.open(io.BytesIO(file.read()))
        processed_img = preprocess_image(img, IMG_SIZE)

        predictions = model.predict(processed_img, verbose=0)
        predicted_idx = np.argmax(predictions)
        confidence = float(predictions[0][predicted_idx])

        if predicted_idx >= len(CLASS_NAMES):
            return jsonify({
                "success": False,
                "error": f"预测索引 {predicted_idx} 超出类别列表范围"
            }), 500

        class_name = CLASS_NAMES[predicted_idx]
        log(f"✅ 识别成功：{class_name}，置信度：{confidence:.2%}")

        return jsonify({
            "success": True,
            "class_name": class_name,
            "confidence": confidence
        })

    except Exception as e:
        log(f"❌ 识别出错：{str(e)}")
        traceback.print_exc(file=sys.stdout)
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": f"服务器内部错误：{str(e)}"
        }), 500


# -------------------------- 静态文件服务 --------------------------
def get_static_folder():
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    else:
        return os.path.abspath('.')


@app.route('/')
def serve_index():
    return send_from_directory(get_static_folder(), 'mushroom.html')


@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory(get_static_folder(), filename)


def open_browser():
    webbrowser.open("http://127.0.0.1:5000")


# -------------------------- 主程序入口（全部包裹在异常捕获中）--------------------------
if __name__ == '__main__':
    try:
        if os.name == 'nt':
            os.system("title 蘑法识界")

        log("=" * 50)
        log("🍄 蘑法识界 正在启动...")
        log("🌐 访问地址：http://127.0.0.1:5000")
        log("📌 关闭本窗口即可退出程序")
        log("=" * 50)

        # 创建数据库表
        log("📦 正在初始化数据库...")
        with app.app_context():
            db.create_all()
        log("✅ 数据库表已创建（或已存在）")

        # 启动浏览器
        Timer(1.5, open_browser).start()

        # 启动 Flask
        log("🚀 正在启动 Flask 服务器...")
        app.run(
            host="0.0.0.0",
            port=5000,
            debug=False,
            use_reloader=False,
            threaded=True
        )
    except Exception as e:
        log(f"\n❌ 启动过程中发生未捕获异常：{e}")
        traceback.print_exc(file=sys.stdout)
        traceback.print_exc()
    finally:
        input("\n⏎ 按回车键退出...")