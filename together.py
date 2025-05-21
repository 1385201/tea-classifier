from flask import Flask, request, jsonify, render_template, send_file
from werkzeug.utils import secure_filename
import os
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'

# 定义预处理函数
def preprocess_image(image_bytes):
    try:
        image = Image.open(BytesIO(image_bytes))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return image
    except:
        print("图像打开出现问题")
        return None

# 定义茶叶种类识别路由
@app.route('/')
def home():
    return render_template('茶叶种类识别.html')

# 定义特定形态识别路由
@app.route('/xingtaishibie')
def xingtaishibie():
    return render_template('特定形态识别.html')

# 定义识别路由
@app.route('/recognize', methods=['POST'])
def recognize():
    recognition_type = request.form.get('recognition_type')
    if recognition_type is None:
        recognition_type = 'default'  # 默认为茶叶种类识别

    if 'image' not in request.files:
        return jsonify({'error': 'no image'}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'no image selected'}), 400

    # 使用BytesIO来处理上传的图片
    image_bytes = file.read()
    image = preprocess_image(image_bytes)
    if image is None:
        return jsonify({'error': '无法正确预处理图像'}), 500

    # 根据识别类型加载不同模型并预测
    if recognition_type == 'single_bud':
        model_path = 'D:/桌面/ultralytics-main/runs/detect/train/weights/best.pt'
    elif recognition_type == 'one_bud_two_leaves':
        model_path = 'D:/桌面/ultralytics-main/runs/detect/train2/weights/best.pt'
    elif recognition_type == 'one_bud_one_leaf':
        model_path = 'D:/桌面/ultralytics-main/runs/detect/train3/weights/best.pt'
    else:  # 默认为茶叶种类识别
        model_path = 'D:/桌面/ultralytics-main/runs/detect/train4/weights/best.pt'

    model = YOLO(model_path)
    results = model(image)

    # 处理预测结果
    predictions = []
    for r in results:
        boxes = r.boxes
        for box in boxes:
            class_id = int(box.cls.item())
            confidence = box.conf.item()
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            if confidence > 0.5:
                predictions.append((class_id, confidence, (x1, y1, x2, y2)))

    # 创建带有预测结果的图片，根据识别类型设置不同颜色的框
    draw = ImageDraw.Draw(image)
    for class_id, confidence, (x1, y1, x2, y2) in predictions:
        if recognition_type == 'single_bud':
            outline_color = "blue"
        elif recognition_type == 'one_bud_two_leaves':
            outline_color = "red"
        elif recognition_type == 'one_bud_one_leaf':
            outline_color = "yellow"
        else:  # 默认为茶叶种类识别
            outline_color = "green"
        draw.rectangle([x1, y1, x2, y2], outline=outline_color, width=3)
        text = f"{class_id} {confidence:.2f}"
        draw.text((x1, y1 - 10), text, fill=outline_color)

    # 保存结果图片并返回
    filename = secure_filename(file.filename)
    result_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    image.save(result_path)
    return send_file(result_path)

if __name__ == '__main__':
    app.run(debug=True)