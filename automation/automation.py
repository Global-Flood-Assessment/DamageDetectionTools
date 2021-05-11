"""
    automation.py 
        --- automate flood classfiction 
"""

prolog="""
**PROGRAM**
    automation.py
      
**PURPOSE**
    Run flood classfiction workflow

**USAGE**
"""
epilog="""
**EXAMPLE**
    python automation.py -g testdata/20210509_one.geojson -s 20210501 -e 20210509 -n test 
"""

# Import modules
import os
import sys
import math
import argparse
from shutil import copyfile

from imagetoolbox import get_footprint, download_imagedata, classification


def _getParser():
    parser = argparse.ArgumentParser(description=prolog,epilog=epilog,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-g', '--geojson', action='store', dest='geojson',required=True,help='region defined by geojson')
    parser.add_argument('-s','--start', action='store', dest='start',required=True,help='start date YYYYMMDD')
    parser.add_argument('-e','--end', action='store', dest='end',required=True,help='end date YYYYMMDD')
    parser.add_argument('-n','--name', action='store', dest='name',required=True,help='project name')
    parser.add_argument('-d','--wkdir', action='store', dest='workdir',required=False,help='working directory')

    # some other possible parameters
    # pfaf_id
    # bbox

    return parser

def run_workflow(paras):
    """ run automation workflow """

    print(paras)

    # step 1: setup working directory
    if paras.workdir is None:
        wkdir = "job_" + paras.name
    else:
        wkdir = paras.worskdir
    
    if not os.path.exists(wkdir):
        os.mkdir(wkdir)
    
    # step 2: processing geojson input
    # copy file to wkdir
    copyfile(paras.geojson, wkdir + os.path.sep + "input.geojson")

    # switch to wkdir
    os.chdir(wkdir)

    # get footprint
    footprint_geojson = get_footprint()

    # step 3: download image data
    cutted_tif = download_imagedata(footprint_geojson,paras.start, paras.end)
    
    # step 4: classification
    classified_img = classification(cutted_tif)

def main():
    
    # Read command line arguments
    parser = _getParser()
    paras = parser.parse_args()
    run_workflow(paras)

if __name__ == '__main__':
    main()