from flask import Flask
from flask import render_template
from flask import request
from predictor import Predictor

# 模型路径
app = Flask(__name__)


@app.route('/')
def category():
    """
    显示文章预测页面
    :return:
    """
    return render_template('News_category.html')


@app.route('/predict', methods=["GET", "POST"])
def predict():
    news_category = Predictor()
    if request.method == "POST":
        news = request.form.get("news")
    else:
        news = request.args.get("news")
    return news_category.predict([news])


if __name__ == '__main__':
    app.run(host='127.0.0.1', debug=True)
