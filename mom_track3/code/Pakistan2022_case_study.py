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

post_date = '20220826'
pre_date = '20220520'
post_end = datetime.datetime.strptime(post_date,"%Y%m%d")
n_days_after = post_end - datetime.timedelta(days=-5)
dc_end = n_days_after.strftime("%Y%m%d")
pre_end = datetime.datetime.strptime(pre_date,"%Y%m%d")
n_days_before = pre_end - datetime.timedelta(days=5)
dc_start = n_days_before.strftime("%Y%m%d")

geojson = gpd.read_file('./20220826_warning.geojson')
watersheds_gdb = "./id_shapefile/Watershed_pfaf_id.shp"
watersheds = gpd.read_file(watersheds_gdb)

for pid in geojson['pfaf_id']:
    geojson_pid = geojson.loc[geojson['pfaf_id'] == pid,'geometry']
    json = geojson_pid.iloc[0]
    a = json.convex_hull
    out_df = watersheds.loc[watersheds['pfaf_id'] == pid]
    if os.path.exists('./shapefile'):
        os.rmdir('./shspefile')
    os.mkdir('./shapefile')
    out_df.to_file('./shapefile/Polygon.shp',driver='ESRI Shapefile',encoding='utf-8')
    s2_classification.classification(a,post_date,dc_end)
    s2_classification.classification(a,dc_start,pre_date)
    dc_start