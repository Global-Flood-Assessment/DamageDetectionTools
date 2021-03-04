import sys
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QComboBox,QLabel,QLineEdit
from folium.plugins import Draw
import folium, io, sys, json
import requests
from bs4 import BeautifulSoup
import urllib
import pandas as pd
import geopandas as gpd
import csv
import os
import shapefile
import datetime
#import cv2
import numpy as np
import zipfile
import tempfile

def Shp2JSON(filename,shp_encoding='utf-8',json_encoding='utf-8'):
    reader = shapefile.Reader(filename,encoding=shp_encoding)
    '''提取所有field部分内容'''
    fields = reader.fields[1:]
    '''提取所有field的名称'''
    field_names = [field[0] for field in fields]
    '''初始化要素列表'''
    buffer = []
    for sr in tqdm(reader.shapeRecords()):
        '''提取每一个矢量对象对应的属性值'''
        record = sr.record
        '''属性转换为列表'''
        record = [r.decode('gb2312','ignore') if isinstance(r, bytes)
                  else r for r in record]
        '''对齐属性与对应数值的键值对'''
        atr = dict(zip(field_names, record))
        '''获取当前矢量对象的类型及矢量信息'''
        geom = sr.shape.__geo_interface__
        '''向要素列表追加新对象'''
        buffer.append(dict(type="Feature",
                           geometry=geom,
                           properties=atr))
    '''写出GeoJSON文件'''
    filename = filename.replace('.shp','.geojson')
    geojson = codecs.open(filename,"w", encoding=json_encoding)
    geojson.write(json.dumps({"type":"FeatureCollection",
                              "features":buffer}) + '\n')
    geojson.close()

# 创建一个 application实例
app = QApplication(sys.argv)
win = QWidget()
win.setWindowTitle('track 3 ui demo')

# 创建一个垂直布局器
layout = QVBoxLayout()
win.setLayout(layout)

# 创建一个按钮去调用function代码
button_track1 = QPushButton('select area from track 1')
button_draw = QPushButton('draw by myself')
button_confirm = QPushButton('start detect')
cb = QComboBox()
cb.move(100, 20)
label_obj1 = QLabel()                # 静态标签
label_obj1.setText(u"welcom to track 3 demo")

def onChanged(text):
    f = open('./shapefile/date.txt','w')
    post_date = text
    post_end = datetime.datetime.strptime(post_date,"%Y%m%d")
    n_days_after = post_end - datetime.timedelta(days=-5)
    pre_end = post_end - datetime.timedelta(days=60)
    dc_end = n_days_after.strftime("%Y%m%d")
    n_days_before = pre_end - datetime.timedelta(days=5)
    dc_start = n_days_before.strftime("%Y%m%d")
    pre_date = dc_end.strftime("%Y%m%d")
    f.write(post_date+' '+dc_end+' '+dc_start+' '+pre_date)

def track1():
    f = open('./shapefile/date.txt','w')
    file_path = os.listdir('./csv/')
    file_path.sort(reverse=True)
    file_path = './csv/'+file_path[0]
    f = csv.reader(open(file_path,'r'))
    for i in f:
        if 'id' in i[0]:
            continue
        cb.addItem(str(i[0])+' '+i[1]+' '+i[2]+' '+i[3]+' '+i[4]+' '+i[5])  #把字符串转化为QListWidgetItem项目对象
    tem = cb.currentText()
    tem = tem.split(' ')
    pid = int(float(tem[0]))
    post_date = tem[-2]
    pre_date = tem[-1]
    post_end = datetime.datetime.strptime(post_date,"%Y%m%d")
    n_days_after = post_end - datetime.timedelta(days=-5)
    dc_end = n_days_after.strftime("%Y%m%d")
    pre_end = datetime.datetime.strptime(pre_date,"%Y%m%d")
    n_days_before = pre_end - datetime.timedelta(days=5)
    dc_start = n_days_before.strftime("%Y%m%d")
    response = requests.get('https://js-157-200.jetstream-cloud.org/ModelofModels/gis_output/flood_warning_'+post_date+'.zip')
    _tmp_file = tempfile.TemporaryFile()
    _tmp_file.write(response.content)
    zf = zipfile.ZipFile(_tmp_file, mode='r')
    if os.path.exists('./zip') == False:
        os.mkdir('./zip')
    for names in zf.namelist():
        f = zf.extract(names, './zip')  # 解压到zip目录文件下
    zf.close()
    shpdata = gpd.GeoDataFrame.from_file('./zip/flood_warning_'+post_date+'.shp')
    shpdata = shpdata.loc[shpdata['pfaf_id'] == pid]
    shpdata.to_file('./shapefile/Polygon.shp',driver='ESRI Shapefile',encoding='utf-8')
    a = shpdata["geometry"]
    a = str(a).split(',')
    a = a[0].split('((')[1]
    a = a.split(' ')
    print([float(a[0]), float(a[1])])
    m = folium.Map(location=[float(a[0]), float(a[1])], tiles="cartodbpositron", zoom_start=7)
    folium.GeoJson(data=shpdata["geometry"]).add_to(m)
    data = io.BytesIO()
    m.save(data, close_file=False)
    w1 = QWebEngineView()
    w1.setHtml(data.getvalue().decode())
    w1.resize(640, 480)
    label_obj1.setText('please choose your interested area')
    layout.replaceWidget(w,w1)
    layout.update()
    win.update()

def draw():
    m = folium.Map(location=[55.8527, 37.5689], zoom_start=13)
    draw = Draw(export=True,filename='footprint.geojson',position='bottomright',draw_options={'polyline':False,'rectangle':True,'polygon':True,'circle':False,'marker':True,'circlemarker':False},edit_options={'edit':False})
    m.add_child(draw)
    data = io.BytesIO()
    m.save(data, close_file=False)
    w1 = QWebEngineView()
    w1.setHtml(data.getvalue().decode())
    w1.resize(640, 480)
    layout.replaceWidget(w,w1)
    label_obj1.setText(u"please input search date like YYYYmmdd:")
    line_edit_obj1 = QLineEdit()        # 单行编辑框
    line_edit_obj1.textChanged[str].connect(onChanged)
    layout.replaceWidget(cb,line_edit_obj1)
    layout.update()
    win.update()

def detect():
    Shp2JSON('./shapefile/Polygon.shp')

# 按钮连接 'complete_name'槽，当点击按钮是会触发信号
button_track1.clicked.connect(track1)
button_draw.clicked.connect(draw)
button_confirm.clicked.connect(detect)
# 把QWebView和button加载到layout布局中
m = folium.Map(location=[55.8527, 37.5689], zoom_start=13)
data = io.BytesIO()
m.save(data, close_file=False)
w = QWebEngineView()
w.setHtml(data.getvalue().decode())
w.resize(640, 480)
layout.addWidget(label_obj1)
layout.addWidget(cb)
layout.addWidget(w)
layout.addWidget(button_track1)
layout.addWidget(button_draw)
layout.addWidget(button_confirm)


# 显示窗口和运行app
win.show()
sys.exit(app.exec_())
