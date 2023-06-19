# import necessary packages
import numpy as np
import pandas as pd
import geopandas as gpd
import pyproj
import folium
from shapely import geometry
import matplotlib.pyplot as plt
import os, shutil
from glob import glob
import xml.dom.minidom
from sentinelsat import SentinelAPI, read_geojson, geojson_to_wkt
from osgeo.gdalconst import *
from geopandas import *
import rasterio as rio
import rasterio.mask
import json
from geopandas import GeoSeries
from osgeo import gdal
from PIL import Image
import matplotlib.image as mpimg
from osgeo import ogr
import subprocess
from collections import Counter
from sklearn.neighbors import KNeighborsClassifier
import joblib
from sklearn.ensemble import RandomForestClassifier

def get_extent(fn):
    ds = gdal.Open(fn)
    rows = ds.RasterYSize
    cols = ds.RasterXSize
    gt = ds.GetGeoTransform()
    minx = gt[0]
    maxy = gt[3]
    maxx = gt[0] + gt[1] * rows
    miny = gt[3] + gt[5] * cols
    return (minx, maxy, maxx, miny)

#read the GeoTIFF file
def read_img(dataset):
    im_width = dataset.RasterXSize
    im_height = dataset.RasterYSize
    im_geotrans = dataset.GetGeoTransform() #affine matrix
    im_proj = dataset.GetProjection() #Map projection information
    im_bands = dataset.RasterCount
    im_data = dataset.ReadAsArray(0,0,im_width,im_height)
    del dataset
    return im_proj,im_geotrans,im_data

def write_img(filename,im_proj,im_geotrans,im_data):
        #type of data in gdal
        #gdal.GDT_Byte,
        #gdal .GDT_UInt16, gdal.GDT_Int16, gdal.GDT_UInt32, gdal.GDT_Int32,
        #gdal.GDT_Float32, gdal.GDT_Float64

        if 'int8' in im_data.dtype.name:
            datatype = gdal.GDT_Byte
        elif 'int16' in im_data.dtype.name:
            datatype = gdal.GDT_UInt16
        else:
            datatype = gdal.GDT_Float32

        if len(im_data.shape) == 3:
            im_height, im_width, im_bands = im_data.shape
        else:
            im_bands, (im_width, im_height) = 1,im_data.shape

        driver = gdal.GetDriverByName("GTiff")
        dataset = driver.Create(filename, im_width, im_height, im_bands, datatype)
        dataset.SetGeoTransform(im_geotrans)
        dataset.SetProjection(im_proj)
        if im_bands == 1:
            dataset.GetRasterBand(1).WriteArray(im_data)
        else:
            for i in range(im_bands):
                dataset.GetRasterBand(i+1).WriteArray(im_data[i])
        del dataset

def get_ndvi(data):
    band8 = data[3]
    band4 = data[0]
    molecule = band8 - band4
    denominator = band8 + band4
    flag = 1-np.where(denominator==0,0,1)
    band = flag*(molecule / (denominator+1e-6))
    return band

def get_ndwi(data):
    band8 = data[3]
    band3 = data[1]
    molecule = band8 - band3
    denominator = band8 + band3
    flag = np.where(denominator==0,0,1)
    band = flag*(molecule / (denominator+1e-6))
    return band
# create_model('mom_track3')
# PATH = 'mom_track3'
# create MD model based on sample tiff file
def create_model(PATH):
    knn = KNeighborsClassifier(n_neighbors=1)
    rf = RandomForestClassifier()
    sample_path = PATH+'/example/S2A_MSIL1C_20200608T161911_N0209_R040_T16PBA_20200608T195512_MDClass.tif'
    sample_db = gdal.Open(sample_path)
    class_count = 6
    sp_proj,sp_geotrans,sp_data = read_img(sample_db)
    Y = sp_data[0,:,:].flatten()
    sample_im_path = PATH+'/example/S2A_MSIL1C_20200608T161911_N0209_R040_T16PBA_20200608T195512.tif'
    sample_im_db = gdal.Open(sample_im_path)
    sp_im_proj,sp_im_geotrans,sp_im_data = read_img(sample_im_db)
    ndvi = get_ndvi(sp_im_data)
    ndwi = get_ndwi(sp_im_data)
    sp_img_data = np.zeros((6,10980,10980))
    for i in range(4) :
        temp = np.max(sp_im_data[i])
        sp_img_data[i,:,:] = sp_im_data[i]/temp
    sp_img_data[4,:,:] = ndvi
    sp_img_data[5,:,:] = ndwi
    X = sp_img_data.transpose((1,2,0))
    X = X.reshape((10980*10980,6))
    knn.fit(X,Y)
    rf.fit(X,Y)
    joblib.dump(knn, PATH+'/saved_model/MD.model')
    joblib.dump(rf, PATH+'/saved_model/RF.model')
    return knn

def load_model(path):
    model = joblib.load(path)
    return model

# detect
def cal_box_new(model,data):
    channel, height, width = data.shape
    X = data.transpose((1,2,0))
    X = X.reshape((height*width,channel))
    prediction= model.predict(X)
    ans = prediction.reshape((height, width))
    return ans

def write_img(filename,im_proj,im_geotrans,im_data):
        #type of data in gdal
        #gdal.GDT_Byte,
        #gdal .GDT_UInt16, gdal.GDT_Int16, gdal.GDT_UInt32, gdal.GDT_Int32,
        #gdal.GDT_Float32, gdal.GDT_Float64
        if 'int8' in im_data.dtype.name:
            datatype = gdal.GDT_Byte
        elif 'int16' in im_data.dtype.name:
            datatype = gdal.GDT_UInt16
        else:
            datatype = gdal.GDT_Float32

        if len(im_data.shape) == 3:
            im_bands, im_height, im_width = im_data.shape
        else:
            im_bands, (im_height, im_width) = 1,im_data.shape

        driver = gdal.GetDriverByName("GTiff")
        dataset = driver.Create(filename, im_width, im_height, im_bands, datatype)
        dataset.SetGeoTransform(im_geotrans)
        dataset.SetProjection(im_proj)
        if im_bands == 1:
            dataset.GetRasterBand(1).WriteArray(im_data)
        else:
            for i in range(im_bands):
                dataset.GetRasterBand(i+1).WriteArray(im_data[i])
        del dataset

start = '20220826'
end = '20220831'
footprint=a

def classification(footprint,start,end):
    # setup API and login info
    api =SentinelAPI('mom_sentinel2', 'sentinel2Download!','https://apihub.copernicus.eu/apihub')
    # search data with some limits like range of date and cloud cover using given footprint.
    products =api.query(footprint,date=(start,end), platformname='Sentinel-2', producttype = 'S2MSI2A', cloudcoverpercentage=(0,30))
    product_name = []
    date = start
    # download searched data to local
    for product in products:
        product_info = api.get_product_odata(product)
        is_online = product_info['Online']
        if is_online:
            print('Product' + product+ ' is online. Starting download.')
            #api.download(product)
            product_name.append(product_info['title'])
        else:
            print('Product' + product + ' is not online.')
            api.trigger_offline_retrieval(product)
    product_name.reverse()
    if len(product_name) == 0:
        return "Sorry, now there is no avaliable data."
    # use terminal quary to translate B2, B3, B4, and B8 as one multiple tif file.
    
    len(product_name)
    for file_name in product_name:
        quary = 'gdalinfo '+file_name+'.zip'
        info = os.popen(quary)
        d = info.read()
        d = d.split('SUBDATASET_1_NAME=')
        d = d[1].split('\n')
        q = 'gdal_translate ' + d[0] + ' '+file_name+'.tif'
        info = os.popen(q)
        d = info.read()
        os.rm(file_name+'.zip')
    minX, maxY, maxX, minY = get_extent(product_name[0])
    if len(product_name)> 1:
        for fn in product_name[1:]:
            minx, maxy, maxx, miny = get_extent(fn)
            minX = min(minX, minx)
            maxY = max(maxY, maxy)
            maxX = max(maxX, maxx)
            minY = min(minY, miny)
    in_ds = gdal.Open(product_name[0])
    gt = in_ds.GetGeoTransform()
    rows = int(int(maxX - minX) / abs(gt[5]))
    cols = int(int(maxY - minY) / gt[1])
    driver = gdal.GetDriverByName("GTiff")
    out_ds = driver.Create('union.tif', rows, cols, 4, gdal.GDT_Byte)
    out_ds.SetProjection(in_ds.GetProjection())
    gt = list(in_ds.GetGeoTransform())
    gt[0], gt[3] = minX, maxY
    out_ds.SetGeoTransform(gt)
    for fn in product_name:
        in_ds = gdal.Open(fn)
        trans = gdal.Transformer(in_ds, out_ds, [])
        success, xyz = trans.TransformPoint(False, 0, 0)
        x, y, z = map(int, xyz)
        data = in_ds.GetRasterBand(1).ReadAsArray()
        for i in range(4):
            out_ds.GetRasterBand(i+1).WriteArray(data[i],x,y)
    del in_ds, out_band, out_ds
    src_path = 'union.tif'
    src = rio.open(src_path)
    shpdata = GeoDataFrame.from_file(PATH+'/shapefile/Polygon.shp')
    out_shpdata = shpdata.copy()
    shpdata=shpdata.to_crs(src.crs)
    features = shpdata.geometry[0].__geo_interface__
    out_image, out_transform = rio.mask.mask(src, [features], crop=True, nodata=src.nodata)
    out_meta = src.meta.copy()
    out_meta.update({"driver": "GTiff",
                     "height": out_image.shape[1],
                     "width": out_image.shape[2],
                     "transform": out_transform})
    band_mask = rasterio.open(PATH+'/geo_tiff/'+date+'cutted.tif', "w", **out_meta)
    band_mask.write(out_image)
    #model = create_model()
    model = load_model(PATH+'/saved_model/MD_model.txt')

    # calculate NDVI & MDWI band for each file
    os.rm('union.tif')
    dataset=gdal.Open(PATH+'/geo_tiff/'+date+'cutted.tif')
    #dataset=gdal.Open('geo_tiff/geo_tiff/20210703.tif')
    im_proj,im_geotrans,im_data = read_img(dataset)
    ndvi = get_ndvi(im_data)
    ndwi = get_ndwi(im_data)
    c,h,w = im_data.shape
    new_data = np.zeros((6,h,w))
    for i in range(4) :
        temp = np.max(im_data[i])
        new_data[i,:,:] = im_data[i]/temp
    new_data[4,:,:] = ndvi
    new_data[5,:,:] = ndwi
    input_pic = new_data
    # detect and save result
    ans = cal_box_new(model,input_pic)

    save_path = PATH+'/output/'
    write_img(save_path+'final_output.tif',im_proj,im_geotrans,ans)
    return "Prediction done."
