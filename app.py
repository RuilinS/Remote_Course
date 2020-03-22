#!flask/bin/python
import os
import numpy as np
from get_digit_info import Solver, Net
from PIL import Image
from werkzeug.utils import secure_filename
from flask import Flask, request, url_for, send_from_directory, jsonify, render_template
import time

# connect to cassandra
import logging
from cassandra.cluster import Cluster
from cassandra.query import SimpleStatement

# 读取IP信息
ip = '127.0.0.1'
with open('./ip.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip()
        if len(line) > 5:
            ip = line
print(ip)

cluster = Cluster(contact_points=[ip], port=9042)
session = cluster.connect()

# load model
solver = Solver()
solver.load_model()
use_db = False

print("load success")

# generate log
log = logging.getLogger()
log.setLevel('INFO')
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s[%(levelname)s] %(name)s: %(message)s"))
log.addHandler(handler)

# create a table
KEYSPACE = "selfkeyspace"


def createKeySpace():
    cluster = Cluster(contact_points=[ip], port=9042)
    session = cluster.connect()

    log.info("Creating keyspace...")
    try:
        session.execute("""
          CREATE KEYSPACE %s
          WITH replication = { 'class': 'SimpleStrategy', 'replication_factor': '2' }
          """ % KEYSPACE)

        log.info("setting keyspace...")
        session.set_keyspace(KEYSPACE)

        log.info("creating table...")
        session.execute("""
          CREATE TABLE selftable (
              selfkey text,
              pic_name text,
              pic_result text,
              PRIMARY KEY (selfkey,pic_name)
          )
          """)
    except Exception as e:
        log.error("Unable to create keyspace")
        log.error(e)


createKeySpace()

# insert data into database
def insert_into_cassandra(var1, var2, var3):
    sql = "INSERT INTO selfkeyspace.selftable (selfkey, pic_name, pic_result) VALUES ('%s', '%s', '%s');""" % (
    var1, var2, var3)
    session.execute(sql)

ALLOWED_EXTENSIONS = set(['png', 'bmp', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.getcwd() + '/upload/'
if os.path.exists(app.config['UPLOAD_FOLDER']) is False:
    os.makedirs(app.config['UPLOAD_FOLDER'])


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


@app.route('/upload/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        im = Image.open(file)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            fi = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            im.save(fi)
            test_result = solver.run_single(fi)
            current = time.strftime("%Y/%m/%d %H:%M:%S")
            insert_into_cassandra(current, filename, test_result)
            print("Test result is {}".format(test_result))
            return render_template('base.html', result=test_result)
    return render_template('base.html')

@app.route('/ocr', methods=['POST'])
def upload():
    result = {}
    result["status_code"] = "200"
    result["status_info"] = "sucess"
    result["digit"] = "-1"
    try:
        file = request.files['file']
        im = Image.open(file)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            fi = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            im.save(fi)
            test_result = solver.run_single(fi)
            current = time.strftime("%Y/%m/%d %H:%M:%S")
            insert_into_cassandra(current, filename, test_result)
            result["digit"] = str(test_result)
    except Exception as e:
        print("Error: ", e)
        result["status_code"] = "500"
        result['status_info'] = str(e)

    return jsonify(result)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=4000, debug=False, threaded=True)


