# from app import app
import os 
import random
from pathlib import Path

from flask import Flask, request, make_response, json, jsonify, redirect, session, render_template, Markup, flash
from flask import url_for, send_from_directory, abort
# from flask_wtf.csrf import validate_csrf 
from werkzeug.utils import secure_filename
from werkzeug.datastructures import FileStorage
# from wtforms import ValidationError
# from flask_ckeditor import CKEditor
# from flask_sqlalchemy import SQLAlchemy
import click

from chat import chat_cls, chat_rawtext

app = Flask(__name__)
# ckeditor = CKEditor(app)
infobox = list() # list of dict
idx = 0

@app.route('/favicon.ico') #浏览器会自动在默认位置寻找favicon.ico
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),                                
    'favicon.ico', mimetype='image/vnd.microsoft.icon')

### 登录页面 ###
@app.route('/')
@app.route('/signin')
def signin():
    return render_template('signin.html')

### 主页面 ###
@app.route('/index',  defaults={'firstwords':"你好，我是小墨"}) #默认参数设置
@app.route('/index/<firstwords>') #参数设置渠道
def index(firstwords):
    return render_template('chatbot.html', firstwords=firstwords)

@app.route('/chat', methods=['POST'])
def chat():
    humanWords = request.get_data()
    humanWords = str(humanWords, encoding='utf-8')
    print(humanWords) # flask的输出会在控制台显示
    bertResp = chat_rawtext(humanWords) #返回一个字典
    infobox.append(bertResp)
    return bertResp['response']


### 后端过程页面 ###
@app.route('/backend') #参数设置渠道
def backend():
    return render_template('backend.html')

@app.route('/show_backend', methods=['POST'])
def show_backend():
    global infobox
    global idx
    bertResp = infobox[idx]
    idx = idx + 1
    return bertResp

