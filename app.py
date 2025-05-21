from flask import Flask, request, jsonify, render_template, send_file, redirect, session, url_for
from werkzeug.utils import secure_filename
import os
import logging
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from pytz import timezone  # 添加时区支持

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SECRET_KEY'] = 'your_secret_key'
db = SQLAlchemy(app)

# 确保上传文件夹存在
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])


# 用户模型
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)


# 使用记录模型（区分种类和形态识别类型）
class UsageRecord(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    upload_time = db.Column(db.DateTime, default=lambda: datetime.now(timezone('Asia/Shanghai')))
    result = db.Column(db.String(200))  # 识别结果文本
    image_data = db.Column(db.LargeBinary)  # 图片二进制数据（存储字节流）
    recognition_type = db.Column(db.String(50))  # 主类型：'variety'（种类）或 'form'（形态）
    detail_type = db.Column(db.String(50))  # 具体类型：如'红茶'、'单芽'等

    def delete(self):
        db.session.delete(self)
        db.session.commit()


# 定义预处理函数
def preprocess_image(image_bytes):
    try:
        image = Image.open(BytesIO(image_bytes))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return image
    except Exception as e:
        logger.error(f"图像预处理错误: {e}")
        return None


# ====================== 模型路径获取函数 ======================
# 形态识别模型路径获取（独立函数）
def get_form_model_path(detail_type):
    form_model_mapping = {
        'single_bud': 'D:/桌面/ultralytics-main/runs/detect/train/weights/best.pt',  # 单芽模型
        'one_bud_one_leaf': 'D:/桌面/ultralytics-main/runs/detect/train2/weights/best.pt',  # 一芽一叶模型
        'one_bud_two_leaves': 'D:/桌面/ultralytics-main/runs/detect/train3/weights/best.pt'  # 一芽两叶模型
    }
    if detail_type in form_model_mapping:
        return form_model_mapping[detail_type]
    raise ValueError(f"无效的形态类型: {detail_type}")

# 种类识别模型路径获取（独立函数）
def get_variety_model_path():
    return 'D:/桌面/ultralytics-main/runs/detect/train5/weights/best.pt'  # 种类识别统一模型路径


# 处理预测结果（根据不同模型使用不同的标签映射）
def process_results(results, recognition_mode, detail_type=None):
    """根据模型类型和识别模式处理预测结果"""
    # 形态识别标签映射（根据不同模型设置）
    form_class_mapping = {
        'single_bud': {0: 'dy-0tea'},          # 单芽模型：class_id=0对应单芽
        'one_bud_one_leaf': {0: 'dy-1tea'},    # 一芽一叶模型：class_id=0对应一芽一叶
        'one_bud_two_leaves': {0: 'dy-2tea'}   # 一芽两叶模型：class_id=0对应一芽两叶
    }

    # 种类识别标签映射
    variety_class_mapping = {
        0: 'GreenTea',
        1: 'RedTea',
        2: 'WhiteTea'
    }

    # 根据识别模式选择映射表
    if recognition_mode == 'variety':
        class_mapping = variety_class_mapping
        logger.info(f"使用种类识别映射表: {class_mapping}，模型类型: {detail_type}")
    else:
        # 形态识别：根据具体模型类型选择映射
        class_mapping = form_class_mapping.get(detail_type, {})
        logger.info(f"使用形态识别映射表（模型: {detail_type}）: {class_mapping}")

    predictions = []
    for r in results:
        for box in r.boxes:
            class_id = int(box.cls.item())
            confidence = box.conf.item()
            if confidence > 0.5:  # 置信度阈值，可调整
                class_name = class_mapping.get(class_id, f'unknown_{class_id}')
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                predictions.append((class_name, confidence, (x1, y1, x2, y2)))
                logger.info(f"模型[{detail_type}]检测到类别ID={class_id}，映射为{class_name}，置信度={confidence:.4f}")

    return predictions


# 添加所有页面路由
@app.route('/')
def home():
    """首页路由"""
    return render_template('index.html')



@app.route('/auth', methods=['GET', 'POST'])
def auth():
    """用户认证路由"""
    if request.method == 'POST':
        action = request.form.get('action')
        if action == 'login':
            username = request.form['username']
            password = request.form['password']
            user = User.query.filter_by(username=username, password=password).first()
            if user:
                session['user_id'] = user.id
                return redirect('/')
            else:
                return render_template('auth.html', error='用户名或密码错误', active_tab='login')
        elif action == 'register':
            username = request.form['username']
            password = request.form['password']
            if User.query.filter_by(username=username).first():
                return render_template('auth.html', error='用户名已存在', active_tab='register')
            new_user = User(username=username, password=password)
            db.session.add(new_user)
            db.session.commit()
            return render_template('auth.html', success='注册成功，请登录', active_tab='login')
    return render_template('auth.html', active_tab='login')


@app.route('/logout')
def logout():
    """退出登录路由"""
    session.pop('user_id', None)
    return redirect('/auth')


@app.route('/usage_records')
def usage_records():
    """使用记录路由（支持筛选种类/形态记录）"""
    if 'user_id' not in session:
        return redirect('/auth')

    filter_type = request.args.get('type', 'all')  # 筛选类型：all/variety/form
    user_id = session['user_id']

    if filter_type == 'variety':
        records = UsageRecord.query.filter_by(
            user_id=user_id,
            recognition_type='variety'
        ).order_by(UsageRecord.upload_time.desc()).all()
    elif filter_type == 'form':
        records = UsageRecord.query.filter_by(
            user_id=user_id,
            recognition_type='form'
        ).order_by(UsageRecord.upload_time.desc()).all()
    else:
        records = UsageRecord.query.filter_by(user_id=user_id).all()

    return render_template('usage_records.html', records=records, filter_type=filter_type)


@app.route('/get_image/<int:record_id>')
def get_image(record_id):
    """获取图片路由"""
    record = UsageRecord.query.get_or_404(record_id)
    return send_file(
        BytesIO(record.image_data),  # 从数据库读取字节数据并转为文件流
        mimetype='image/jpeg',
        as_attachment=False
    )


@app.route('/delete_record/<int:record_id>', methods=['POST'])
def delete_record(record_id):
    """删除记录路由"""
    if 'user_id' not in session:
        return jsonify({'error': '未登录'}), 401
    record = UsageRecord.query.get_or_404(record_id)
    if record.user_id != session['user_id']:
        return jsonify({'error': '权限不足'}), 403

    try:
        record.delete()
        return jsonify({'success': True, 'message': '记录已删除'})
    except Exception as e:
        logger.error(f"删除记录失败: {e}")
        return jsonify({'success': False, 'error': '删除记录失败'}), 500


# 形态识别页面路由
@app.route('/xingtaishibie')
def xingtaishibie():
    """特定形态识别页面路由"""
    if 'user_id' in session:
        return render_template('特定形态识别.html')
    else:
        return redirect('/auth')


# 种类识别页面路由
@app.route('/zhongleishibie')
def zhongleishibie():
    """茶叶种类识别页面路由"""
    if 'user_id' in session:
        return render_template('茶叶种类识别.html')
    else:
        return redirect('/auth')

# 茶叶时期识别页面路由
@app.route('/shiqishibie')
def tea_period_recognition():
    """茶叶时期识别页面路由"""
    if 'user_id' in session:
        return render_template('茶叶时期识别.html')  # 渲染对应的HTML页面
    else:
        return redirect('/auth')  # 未登录则重定向到认证页面

@app.route('/recognize', methods=['POST'])
def recognize():
    try:
        if 'user_id' not in session:
            logger.warning("未登录用户尝试访问识别接口")
            return jsonify({'error': '未登录'}), 401

        if 'image' not in request.files:
            logger.warning("未上传图片")
            return jsonify({'error': '未上传图片'}), 400

        file = request.files['image']
        if file.filename == '':
            logger.warning("未选择图片")
            return jsonify({'error': '未选择图片'}), 400

        # 获取前端传递的识别类型
        recognition_type = request.form.get('recognition_type', 'tea_variety')
        logger.info(f"收到识别请求，类型: {recognition_type}")

        # 调整有效类型列表，包含 tea_variety
        valid_form_types = ['single_bud', 'one_bud_one_leaf', 'one_bud_two_leaves']
        valid_variety_types = ['tea_variety', 'green_tea', 'red_tea', 'white_tea']  # 添加 tea_variety

        # 确定识别模式和详细类型
        if recognition_type in valid_form_types:
            recognition_mode = 'form'
            detail_type = recognition_type
        elif recognition_type in valid_variety_types:
            recognition_mode = 'variety'
            detail_type = recognition_type
        else:
            logger.warning(f"无效的识别类型参数: {recognition_type}，使用默认值")
            recognition_mode = 'variety'
            detail_type = 'unknown'  # 明确标记为未知类型

        logger.info(f"识别模式: {recognition_mode}, 详细类型: {detail_type}")

        # 读取并预处理图片
        image_bytes = file.read()
        image = preprocess_image(image_bytes)
        if image is None:
            logger.error("图片预处理失败")
            return jsonify({'error': '无法处理图片'}), 400

        # 根据模式调用不同的模型路径获取函数
        if recognition_mode == 'form':
            model_path = get_form_model_path(detail_type)
            logger.info(f"调用形态识别模型: {model_path} 用于 {detail_type}")
        else:
            model_path = get_variety_model_path()
            logger.info(f"调用种类识别模型: {model_path} 用于 {detail_type}")

        logger.info(f"加载模型: {model_path}")
        if not os.path.exists(model_path):
            logger.error(f"模型文件不存在: {model_path}")
            return jsonify({'error': '模型文件不存在'}), 500

        model = YOLO(model_path)

        # 执行预测
        results = model(image)
        logger.info(f"预测完成，检测到 {len(results[0].boxes)} 个目标")

        # 记录每个检测结果的详细信息
        for i, box in enumerate(results[0].boxes):
            class_id = int(box.cls.item())
            confidence = box.conf.item()
            logger.info(f"检测结果 {i + 1}: 类别ID={class_id}, 置信度={confidence:.4f}")

        # 处理预测结果
        predictions = process_results(results, recognition_mode, detail_type)

        # 转换为前端友好的显示名称
        if recognition_mode == 'form':
            display_mapping = {
                'dy-0tea': '单芽',
                'dy-1tea': '一芽一叶',
                'dy-2tea': '一芽两叶'
            }
            result_text = '; '.join([
                f"{display_mapping.get(cls, cls)} {conf:.2f}"
                for cls, conf, _ in predictions
            ])
        else:
            # 种类识别的显示映射
            variety_display_mapping = {
                'GreenTea': '绿茶',
                'RedTea': '红茶',
                'WhiteTea': '白茶'
            }
            result_text = '; '.join([
                f"{variety_display_mapping.get(cls, cls)} {conf:.2f}"
                for cls, conf, _ in predictions
            ])

        logger.info(f"识别结果: {result_text}")

        # 绘制识别结果（优化字体加载和标签显示）
        draw = ImageDraw.Draw(image)

        # 优化字体加载逻辑，增加对中文的支持
        font = None
        font_paths = [
            "simhei.ttf",  # 当前目录下的黑体
            "C:/Windows/Fonts/simhei.ttf",  # Windows系统字体
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",  # Linux系统字体
            "/System/Library/Fonts/PingFang.ttc",  # macOS系统字体
            "arial.ttf"  # 备选英文字体
        ]

        # 尝试加载可用字体
        for font_path in font_paths:
            if os.path.exists(font_path):
                try:
                    font = ImageFont.truetype(font_path, size=20)  # 减小字体大小，避免超出图片边界
                    logger.info(f"成功加载字体: {font_path}")
                    break
                except Exception as e:
                    logger.warning(f"尝试加载字体 {font_path} 失败: {str(e)}")

        # 如果没有找到任何字体，使用默认字体
        if font is None:
            font = ImageFont.load_default()
            logger.warning("没有找到支持中文的字体，将使用默认字体，可能无法正确显示中文")

        # 设置边框和文本颜色
        outline_color = "blue" if recognition_mode == 'variety' else "red"
        text_color = "white"  # 使用白色文本，提高对比度

        # 绘制识别框和标签
        for class_name, confidence, (x1, y1, x2, y2) in predictions:
            # 获取显示文本
            if recognition_mode == 'form':
                display_text = display_mapping.get(class_name, class_name)
            else:
                display_text = variety_display_mapping.get(class_name, class_name)

            # 构建完整标签文本
            label_text = f"{display_text} {confidence:.2f}"

            # 计算文本尺寸（兼容新旧Pillow版本）
            try:
                # 新方法（Pillow 8.0.0+）
                bbox = draw.textbbox((0, 0), label_text, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
            except AttributeError:
                # 旧方法（Pillow < 8.0.0）
                text_width, text_height = draw.textsize(label_text, font=font)

            # 确保文本绘制位置不会超出图像边界
            x_text = x1
            y_text = max(0, y1 - text_height)  # 防止文本绘制到图像外部

            # 绘制文本背景矩形，提高可读性
            draw.rectangle([x_text, y_text, x_text + text_width, y_text + text_height], fill=outline_color)

            # 绘制识别框
            draw.rectangle([x1, y1, x2, y2], outline=outline_color, width=3)  # 减小边框宽度，更美观

            # 绘制标签文本
            draw.text((x_text, y_text), label_text, fill=text_color, font=font)

        # 保存图片字节数据
        img_byte_arr = BytesIO()
        image.save(img_byte_arr, format='JPEG')
        image_data = img_byte_arr.getvalue()

        # 保存记录到数据库（优化detail_type存储）
        if recognition_mode == 'variety' and detail_type == 'tea_variety':
            # 从识别结果中提取第一个类别作为detail_type
            if predictions:
                first_class = predictions[0][0]  # 获取第一个预测的类别
                detail_type = {
                    'GreenTea': 'green_tea',
                    'RedTea': 'red_tea',
                    'WhiteTea': 'white_tea'
                }.get(first_class, detail_type)  # 如果映射不存在，则保持原detail_type
                logger.info(f"从识别结果中提取detail_type: {detail_type}")

        new_record = UsageRecord(
            user_id=session['user_id'],
            result=result_text,
            image_data=image_data,
            recognition_type=recognition_mode,
            detail_type=detail_type
        )

        db.session.add(new_record)
        db.session.commit()
        logger.info(f"记录保存成功，ID: {new_record.id}，detail_type: {detail_type}")

        # 返回处理后的图片流
        img_byte_arr.seek(0)
        return send_file(img_byte_arr, mimetype='image/jpeg')

    except Exception as e:
        # 回滚数据库事务
        if db.session.is_active:
            db.session.rollback()

        logger.exception("识别过程发生错误")
        return jsonify({'error': f'处理请求时发生错误: {str(e)}'}), 500


if __name__ == '__main__':
    with app.app_context():
        db.create_all()  # 自动创建表（首次运行需删除旧users.db）
    app.run(debug=True, port=5000)