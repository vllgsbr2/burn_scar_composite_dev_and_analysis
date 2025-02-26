#!/usr/bin/env python

# script supports either python2 or python3
#
# Attempts to do HTTP Gets with urllib2(py2) urllib.requets(py3) or subprocess
# if tlsv1.1+ isn't supported by the python ssl module
#
# Will download csv or json depending on which python module is available
#

from __future__ import (division, print_function, absolute_import, unicode_literals)

import argparse
import os
import os.path
import shutil
import sys

try:
    from StringIO import StringIO   # python2
except ImportError:
    from io import StringIO         # python3


################################################################################


USERAGENT = 'tis/download.py_1.0--' + sys.version.replace('\n','').replace('\r','')


def geturl(url, token=None, out=None):
    headers = { 'user-agent' : USERAGENT }
    if not token is None:
        headers['Authorization'] = 'Bearer ' + token
    try:
        import ssl
        CTX = ssl.SSLContext(ssl.PROTOCOL_TLSv1_2)
        if sys.version_info.major == 2:
            import urllib2
            try:
                fh = urllib2.urlopen(urllib2.Request(url, headers=headers), context=CTX)
                if out is None:
                    return fh.read()
                else:
                    shutil.copyfileobj(fh, out)
            except urllib2.HTTPError as e:
                print('HTTP GET error code: %d' % e.code(), file=sys.stderr)
                print('HTTP GET error message: %s' % e.message, file=sys.stderr)
            except urllib2.URLError as e:
                print('Failed to make request: %s' % e.reason, file=sys.stderr)
            return None

        else:
            from urllib.request import urlopen, Request, URLError, HTTPError
            try:
                fh = urlopen(Request(url, headers=headers), context=CTX)
                if out is None:
                    return fh.read().decode('utf-8')
                else:
                    shutil.copyfileobj(fh, out)
            except HTTPError as e:
                print('HTTP GET error code: %d' % e.code(), file=sys.stderr)
                print('HTTP GET error message: %s' % e.message, file=sys.stderr)
            except URLError as e:
                print('Failed to make request: %s' % e.reason, file=sys.stderr)
            return None

    except AttributeError:
        # OS X Python 2 and 3 don't support tlsv1.1+ therefore... curl
        import subprocess
        try:
            args = ['curl', '--fail', '-sS', '-L', '--get', url]
            for (k,v) in headers.items():
                args.extend(['-H', ': '.join([k, v])])
            if out is None:
                # python3's subprocess.check_output returns stdout as a byte string
                result = subprocess.check_output(args)
                return result.decode('utf-8') if isinstance(result, bytes) else result
            else:
                subprocess.call(args, stdout=out)
        except subprocess.CalledProcessError as e:
            print('curl GET error message: %' + (e.message if hasattr(e, 'message') else e.output), file=sys.stderr)
        return None



################################################################################


DESC = "This script will recursively download all files if they don't exist from a LAADS URL and stores them to the specified path"


def sync(src, dest, tok):
    '''synchronize src url with dest directory'''
    try:
        import csv
        files = [ f for f in csv.DictReader(StringIO(geturl('%s.csv' % src, tok)), skipinitialspace=True) ]
    except ImportError:
        import json
        files = json.loads(geturl(src + '.json', tok))

    # use os.path since python 2/3 both support it while pathlib is 3.4+
    for f_num, f in enumerate(files):
        # currently we use filesize of 0 to indicate directory
        filesize = int(f['size'])
        path = os.path.join(dest, f['name'])
        url = src + '/' + f['name']
        if filesize == 0:
            try:
                print('creating dir:', path)
                os.mkdir(path)
                sync(src + '/' + f['name'], path, tok)
            except IOError as e:
                print("mkdir `%s': %s" % (e.filename, e.strerror), file=sys.stderr)
                sys.exit(-1)
        else:
            try:
                if not os.path.exists(path):
                    # x = ' {:4d} downloading: {}'.format(f_num, path)
                    # print(x)
                    print('downloading: ' , path)
                    with open(path, 'w+b') as fh:
                        geturl(url, tok, fh)
                else:
                    print('skipping file on disk: ', path)
            except IOError as e:
                print("open `%s': %s" % (e.filename, e.strerror), file=sys.stderr)
                sys.exit(-1)
    return 0


def _main(argv):
    parser = argparse.ArgumentParser(prog=argv[0], description=DESC)
    parser.add_argument('-s', '--source', dest='source', metavar='URL', help='Recursively download files at URL', required=True)
    parser.add_argument('-d', '--destination', dest='destination', metavar='DIR', help='Store directory structure in DIR', required=True)
    parser.add_argument('-t', '--token', dest='token', metavar='TOK', help='Use app token TOK to authenticate', required=True)
    args = parser.parse_args(argv[1:])
    if not os.path.exists(args.destination):
        os.makedirs(args.destination)
    return sync(args.source, args.destination, args.token)


if __name__ == '__main__':
    try:
        sys.exit(_main(sys.argv))
    except KeyboardInterrupt:
        sys.exit(-1)

    token                 = 'eyJ0eXAiOiJKV1QiLCJvcmlnaW4iOiJFYXJ0aGRhdGEgTG9naW4iLCJzaWciOiJlZGxqd3RwdWJrZXlfb3BzIiwiYWxnIjoiUlMyNTYifQ.eyJ0eXBlIjoiVXNlciIsInVpZCI6InZsbGdzYnIyIiwiZXhwIjoxNzM0ODg2Mzc3LCJpYXQiOjE3Mjk3MDIzNzcsImlzcyI6Imh0dHBzOi8vdXJzLmVhcnRoZGF0YS5uYXNhLmdvdiJ9.I6RpfWYvXY6TTudGCscpbEGWTxFBynjqrR_wnUWtFohMtNlB7HugxmzsrQ0jv3UtXxi1jmPtlX140puHAPcQhfL6XKqSWfySHKzuxlO6dfZvz0ZoGuepsCSEns5nM9O_J0yp3SOR431soxDw1JSJ77AfjVQ-K786I9DjhgNL4LHsE1UwH8F19uzNxQa2LiavoMnI7344ONs4NfpsoP8bnGg3VjML5z54gTW5TMW4DwJ9AdbpaTJjfhCXEG4CxffKK8-2JPaBReuLmZ89oDv8PU_ieh8bKB_I4wL5Lc_SdQd9_QkR6til0xaHMR99GbYcXSlCw4xhZanJcpMkJDHuDg' 
    destination_directory = '/scratch/zt1/project/vllgsbr2-prj/raw_data_burnscar/data/noaa_20_viirs/2024_fire_season'
    source_url            = 'https://ladsweb.modaps.eosdis.nasa.gov/archive/orders/502165538/'


    #order_2020_vj103 = '502165538/' # donzo
    #order_2020_vj109 = '502165537/' # donzo
    #order_2021_vj103 = '502165516/' # donzo
    #order_2021_vj109 = '502165515/' # donzo

    #order_park_fire_VJ109 = '502265239/' donzo
    #order_park_fire_VJ103 = '502265240/' donzo
    #_main()

    #order_07012024_08252024_VJ103_VJ109 = '502293922/' not donzo
    #order_07012024_08252024_VJ114_VJ115 = '502293921/' not donzo

    #LA_fires_12_20_24_01_10_25_VJ109_VJ103 = 502321825
    #LA_fires_12_20_24_01_10_25_VJ114_VJ115 = 502321824

    '''
    how to run

    python laads-data-download.py -s <url of order> -d <destination directory> -t <token>

    to generate a token, log into the NASA LAADS DAAC and goto find data and then goto orders, in 
    top right of screen click login drop down menu. There, hit generate token.

    i.e.

    python laads-data-download.py -s https://ladsweb.modaps.eosdis.nasa.gov/archive/orders/502141057/ -d /scratch/zt1/project/vllgsbr2-prj/burnscar_project/data/noaa_20_vj109_files -t eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJlbWFpbF9hZGRyZXNzIjoiamF2aWVyLmEudmlsbGVnYXNicmF2b0Bub2FhLmdvdiIsImlzcyI6IkFQUyBPQXV0aDIgQXV0aGVudGljYXRvciIsImlhdCI6MTcxMDk1NjkxMSwibmJmIjoxNzEwOTU2OTExLCJleHAiOjE4Njg2MzY5MTEsInVpZCI6InZsbGdzYnIyIiwidG9rZW5DcmVhdG9yIjoidmxsZ3NicjIifQ.hK7ULSDjMBLm-uHTPApUj7OBHOGFZQA0KsxtzcriQUA
    '''
