## Automation of Image Classification  
### automation.py  
**paramters:**  
```
  -g GEOJSON, --geojson GEOJSON
                        region defined by geojson
  -s START, --start START
                        start date YYYYMMDD
  -e END, --end END     end date YYYYMMDD
  -n NAME, --name NAME  project name
  
  optional:
  -d WORKDIR, --wkdir WORKDIR  working directory
```
**usage:**
```
python automation.py -g testdata/20210509_one.geojson -s 20210501 -e 20210509 -n test 
```

**output:**   
default folder: job_[name], job_test in this example   
```
classfied.tif  -- classified image
cutted.tif     -- image outlined by footprint
union.tif      -- image mosaic of all download images
footprint.geojson  -- convexhull of input regions
input.geojson      -- input regions 

-- converted geotiffs
S2B_MSIL2A_20210508T002709_N0300_R016_T54HXG_20210508T020135.tif 
S2B_MSIL2A_20210508T002709_N0300_R016_T54HYG_20210508T020135.tif

-- downloaded data
S2B_MSIL2A_20210508T002709_N0300_R016_T54HXG_20210508T020135.zip
S2B_MSIL2A_20210508T002709_N0300_R016_T54HYG_20210508T020135.zip
```
3.9GB in total

**testing environments:**  
Install [Miniconda Python 3.8](https://docs.conda.io/en/latest/miniconda.html)      
Run ```conda env create -f environment.yml``` to install all neccesary packages

**known issues and todos**  
* environment.yml needs clean-up
* download offline data (older than 30 days) is not supported  
* add logging and generate job summary
* generate more GIS outputs 
