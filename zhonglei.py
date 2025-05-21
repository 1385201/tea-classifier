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
        if image.mode!= 'RGB':
            image = image.convert('RGB')
        return image
    except:
        print("图像打开出现问题")
        return None


@app.route('/')
def home():
    return render_template('茶叶种类识别.html')


@app.route('/recognize', methods=['POST'])
def recognize():
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

    # 加载茶叶种类识别模型并预测
    model_path = 'D:/桌面/ultralytics-main/runs/detect/train4/weights/best.pt'
    model = YOLO(model_path)

    results = model(image)

    # 获取模型中的类别名称
    class_names = model.names

    # 处理预测结果
    predictions = []
    for r in results:
        boxes = r.boxes
        for box in boxes:
            class_id = int(box.cls.item())
            confidence = box.conf.item()
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            if confidence > 0.5:
                class_name = class_names[class_id]
                predictions.append((class_name, confidence, (x1, y1, x2, y2)))

    # 创建带有预测结果的图片，进一步调整文字大小和加粗
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("arialbd.ttf", size=24)  # 使用加粗字体（arialbd.ttf表示加粗的Arial字体），并适当增大字号为24，可按需调整
    for class_name, confidence, (x1, y1, x2, y2) in predictions:
        outline_color = "blue"
        draw.rectangle([x1, y1, x2, y2], outline=outline_color, width=5)
        text = f"{class_name} {confidence:.2f}"
        draw.text((x1, y1 - 10), text, fill=outline_color, font=font)

    # 保存结果图片并返回
    filename = secure_filename(file.filename)
    result_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    image.save(result_path)
    return send_file(result_path)


if __name__ == '__main__':
    app.run(debug=True)