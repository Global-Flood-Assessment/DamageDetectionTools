"""
    imagetoolbox.py 
        --- toolbox for image processing
"""
import os
import numpy as np
import geopandas as gpd
from sentinelsat import SentinelAPI, read_geojson, geojson_to_wkt
from gdalconst import *
from osgeo import gdal
import rasterio as rio
import rasterio.mask

def get_extent(fn):
    """get raster extemt"""

    ds = gdal.Open(fn)
    rows = ds.RasterYSize
    cols = ds.RasterXSize
    gt = ds.GetGeoTransform()
    minx = gt[0]
    maxy = gt[3]
    maxx = gt[0] + gt[1] * rows
    miny = gt[3] + gt[5] * cols
    return (minx, maxy, maxx, miny)

def load_model(path):
    model_file = open(path,'r')
    model = []
    for line in model_file:
        a = line.strip('\n').strip('[').strip(']').split(' ')
        #print(a)
        model.append(list(map(float,line.strip('\n').strip('[').strip(']').split(' '))))
    #print(model)

    return model

def get_ndvi(data):
    band8 = data[3]
    band4 = data[0]
    molecule = band8 - band4
    denominator = band8 + band4
    band = molecule / denominator
    band[band > 1] = -999
    band[band < -1] = -999
    return band

def get_ndwi(data):
    band8 = data[3]
    band3 = data[1]
    molecule = band8 - band3
    denominator = band8 + band3
    band = molecule / denominator
    band[band > 1] = -999
    band[band < -1] = -999
    return band

# detect
def cal_box_new(model,data):
    channel, height, width = data.shape
    ans_pic = []
    for k in range(len(model)):
        flag = model[k]
        flag_map = np.tile(flag,(height,width,1))
        flag_map = flag_map.transpose((2,0,1))
        differ = np.sqrt(sum(np.power((data - flag_map), 2)))
        ans_pic.append(differ)
    ans_pic = np.asarray(ans_pic)
    ans=np.argmin(ans_pic,axis=0)
    return ans

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
            im_bands, (im_height, im_width) = 1,im_data.shape
        print(im_bands,im_width,im_height)
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

def get_footprint(input_geojson = "input.geojson", footprint_geojson = "footprint.geojson"):
    """ read geojson file and generate footprint """
    
    raw_gpd = gpd.read_file(input_geojson)

    # generate convex hull
    exterior = gpd.GeoDataFrame(geometry = gpd.GeoSeries(raw_gpd.unary_union).exterior.convex_hull)
    exterior.crs = 'EPSG:4326'
    exterior.to_file(footprint_geojson, driver='GeoJSON')
    
    return footprint_geojson

def download_imagedata(footprint_geojson, startdate, enddate):
    """ download image data with SentinelAPI """
    
    # .netrc method is not working 
    api =SentinelAPI('flooda31', 'floodproject','https://scihub.copernicus.eu/dhus')
    footprint = geojson_to_wkt(read_geojson(footprint_geojson))
    products =api.query(footprint,date=(startdate,enddate), platformname='Sentinel-2', producttype = 'S2MSI2A', cloudcoverpercentage=(0,30))
    
    product_name = []
    date = startdate

    # download searched data to local
    for product in products:
        product_info = api.get_product_odata(product)
        is_online = product_info['Online']

        if is_online:
            product_name.append(product_info['title'])
            api.download(product)
        else:
            # offline data issue: 
            # https://scihub.copernicus.eu/userguide/LongTermArchive
            # https://sentinelsat.readthedocs.io/en/latest/api_overview.html#lta-products
            # may try asf
            print('Product {} is not online.'.format(product))

    product_name.reverse()
    
    # use terminal quary to translate B2, B3, B4, and B8 as one multiple tif file.
    for file_name in product_name:
        print(file_name)
        quary = 'gdalinfo ' + file_name + ".zip"
        info = os.popen(quary)
        d = info.read()
        d = d.split('SUBDATASET_1_NAME=')
        d = d[1].split('\n')
        q = 'gdal_translate ' + d[0] + ' '+file_name+'.tif'
        info = os.popen(q)
        d = info.read()

    minX, maxY, maxX, minY = get_extent(product_name[0]+".tif")
    if len(product_name)> 1:
        for fn in product_name[1:]:
            minx, maxy, maxx, miny = get_extent(fn + ".tif")
            minX = min(minX, minx)
            maxY = max(maxY, maxy)
            maxX = max(maxX, maxx)
            minY = min(minY, miny)

    in_ds = gdal.Open(product_name[0]+".tif")
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
        in_ds = gdal.Open(fn + ".tif")
        trans = gdal.Transformer(in_ds, out_ds, [])
        success, xyz = trans.TransformPoint(False, 0, 0)
        x, y, z = map(int, xyz)
        
        for i in range(4):
            data = in_ds.GetRasterBand(i+1).ReadAsArray()
            out_ds.GetRasterBand(i+1).WriteArray(data,x,y)
    del in_ds, out_ds
    
    # cut image
    src_path = 'union.tif'
    src = rio.open(src_path)
    
    shpdata = gpd.read_file(footprint_geojson)
    out_shpdata = shpdata.copy()
    shpdata=shpdata.to_crs(src.crs)
    
    features = shpdata.geometry[0].__geo_interface__
    out_image, out_transform = rio.mask.mask(src, [features], crop=True, nodata=src.nodata)
    out_meta = src.meta.copy()
    out_meta.update({"driver": "GTiff",
                     "height": out_image.shape[1],
                     "width": out_image.shape[2],
                     "transform": out_transform})
    cutted_tif = "cutted.tif"
    band_mask = rio.open(cutted_tif, "w", **out_meta)
    band_mask.write(out_image)

    return cutted_tif

def classification(img):
    """classify image"""
    
    model = load_model('../saved_model/MD_model.txt')
        # calculate NDVI & MDWI band for each file
    
    dataset=gdal.Open(img)
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
    outimg="classfied.tif"
    print(ans.shape)
    write_img(outimg,im_proj,im_geotrans,ans)
    
    return outimg

def main():
    
    # test 
    os.chdir("job_test")
    #footprint_geojson = get_footprint()
    #cutted_tif = download_imagedata(footprint_geojson,"20210501","20210509")
    cutted_tif = "cutted.tif"
    classification(cutted_tif)

if __name__ == '__main__':
    main()
