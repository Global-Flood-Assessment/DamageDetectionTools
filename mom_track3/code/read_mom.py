import requests
from bs4 import BeautifulSoup
import csv
import urllib
import pandas as pd
import geopandas as gpd
import datetime
import cv2
import numpy as np
from code import s2_classification 
import zipfile
import tempfile
import os

# get all csv files
url = 'https://js-157-200.jetstream-cloud.org/ModelofModels/flood_severity/'
html = requests.get(url)
soup = BeautifulSoup(html.content, 'html.parser')
links = soup.find_all('a')
csv_file_list = []
for link in links:
    a = link.get("href")
    if 'Final' in a:
        csv_file_list.append(url+a)
csv_file_list.reverse()
s = pd.DataFrame({'pfaf_id':[],'Modified_Severity':[],'post_date':[],'pre_date':[]})
# check each one to find out top 10 serious flooding area
for first in range(7):
    df = pd.read_csv(csv_file_list[first])
    data=df.sort_values(by="Severity" , ascending=False)
    date = csv_file_list[0].split('_')
    date = date[-1].replace('.csv','')
    data = data.iloc[[i for i in range(10)],[0,-3]]
    data['post_date']=date
    s = s.append(data)
    days = 1
    while data.empty is False :
        now = datetime.datetime.strptime(date,"%Y%m%d")
        n_days_before = now - datetime.timedelta(days=days)
        dc = n_days_before.strftime("%Y%m%d")
        pre_csv = csv_file_list[0].replace(date,dc)
        df = pd.read_csv(pre_csv)
        check = df.loc[df['pfaf_id'].isin(data.iloc[:,0])]
        if check.empty:
            s.loc[s['pfaf_id'].isin(data.iloc[:,0]),'pre_date']=dc
            break
        check = check.loc[check['Alert'] != 'Warning']
        if days >= 60 :
            s.loc[s['pfaf_id'].isin(data.iloc[:,0]),'pre_date']=dc
            break
        if check.empty :
            days += 1
        else:
            s.loc[s['pfaf_id'].isin(check.iloc[:,0]),'pre_date']=dc
            data = data.loc[~data['pfaf_id'].isin(check.iloc[:,0])]

s = s.sort_values(by="Modified_Severity",ascending=False)
s.to_csv('../csv_file/most_severe_flood_'+date+'.csv',index=False)
for row in s.itertuples():
    pid = int(row[1])
    post_date = row[3]
    pre_date = row[5]
    post_end = datetime.datetime.strptime(post_date,"%Y%m%d")
    n_days_after = post_end - datetime.timedelta(days=-5)
    dc_end = n_days_after.strftime("%Y%m%d")
    pre_end = datetime.datetime.strptime(pre_date,"%Y%m%d")
    n_days_before = pre_end - datetime.timedelta(days=5)
    dc_start = n_days_before.strftime("%Y%m%d")
    geojson = gpd.read_file('https://js-157-200.jetstream-cloud.org/ModelofModels/gis_output/flood_warning_'+post_date+'.geojson')
    geojson = geojson.loc[geojson['pfaf_id'] == pid,'geometry']
    json = geojson.iloc[0]
    a = json.convex_hull
    response = requests.get('https://js-157-200.jetstream-cloud.org/ModelofModels/gis_output/flood_warning_'+post_date+'.zip')
    _tmp_file = tempfile.TemporaryFile()
    _tmp_file.write(response.content)
    zf = zipfile.ZipFile(_tmp_file, mode='r')
    if os.path.exists('../zip'):
        os.rmdir('../zip')
    os.mkdir('../zip')
    for names in zf.namelist():
        f = zf.extract(names, '../zip')  # 解压到zip目录文件下
    zf.close()
    os.rmdir('../zip')
    shpdata = gpd.GeoDataFrame.from_file('../zip/flood_warning_'+post_date+'.shp')
    shpdata = shpdata.loc[shpdata['pfaf_id'] == pid]
    shpdata.to_file('../shapefile/Polygon.shp',driver='ESRI Shapefile',encoding='utf-8')
    s2_classification.classification(a,post_date,dc_end)
    s2_classification.classification(a,dc_start,pre_date)
