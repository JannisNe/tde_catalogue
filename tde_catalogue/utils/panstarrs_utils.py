import sys, os, re, json, io, tqdm
import numpy as np
from urllib.parse import quote as urlencode
import http.client as httplib
from astropy.table import Table
import requests
from PIL import Image
from io import BytesIO
import pandas as pd
import matplotlib.pyplot as plt

# get the WSID and password if not already defined
import getpass
if not os.environ.get('CASJOBS_WSID'):
    os.environ['CASJOBS_WSID'] = input('Enter Casjobs WSID:')
if not os.environ.get('CASJOBS_PW'):
    os.environ['CASJOBS_PW'] = getpass.getpass('Enter Casjobs password:')

from tde_catalogue import main_logger


logger = main_logger.getChild(__name__)
crossmatch_url = 'https://catalogs.mast.stsci.edu/api/v0.1/panstarrs/mean/crossmatch/upload.csv'


####################################
# START define some util functions #
####################################


def getimages(ra, dec, size=240, filters="grizy"):
    """Query ps1filenames.py service to get a list of images

    ra, dec = position in degrees
    size = image size in pixels (0.25 arcsec/pixel)
    filters = string with filters to include
    Returns a table with the results
    """

    service = "https://ps1images.stsci.edu/cgi-bin/ps1filenames.py"
    url = ("{service}?ra={ra}&dec={dec}&size={size}&format=fits"
           "&filters={filters}").format(**locals())
    table = Table.read(url, format='ascii')
    return table


def geturl(ra, dec, size=240, output_size=None, filters="grizy", format="jpg", color=False):
    """Get URL for images in the table

    ra, dec = position in degrees
    size = extracted image size in pixels (0.25 arcsec/pixel)
    output_size = output (display) image size in pixels (default = size).
                  output_size has no effect for fits format images.
    filters = string with filters to include
    format = data format (options are "jpg", "png" or "fits")
    color = if True, creates a color image (only for jpg or png format).
            Default is return a list of URLs for single-filter grayscale images.
    Returns a string with the URL
    """

    if color and format == "fits":
        raise ValueError("color images are available only for jpg or png formats")
    if format not in ("jpg", "png", "fits"):
        raise ValueError("format must be one of jpg, png, fits")
    table = getimages(ra, dec, size=size, filters=filters)
    url = ("https://ps1images.stsci.edu/cgi-bin/fitscut.cgi?"
           "ra={ra}&dec={dec}&size={size}&format={format}").format(**locals())
    if output_size:
        url = url + "&output_size={}".format(output_size)
    # sort filters from red to blue
    flist = ["yzirg".find(x) for x in table['filter']]
    table = table[np.argsort(flist)]
    if color:
        if len(table) > 3:
            # pick 3 filters
            table = table[[0, len(table) // 2, len(table) - 1]]
        for i, param in enumerate(["red", "green", "blue"]):
            url = url + "&{}={}".format(param, table['filename'][i])
    else:
        urlbase = url + "&red="
        url = []
        for filename in table['filename']:
            url.append(urlbase + filename)
    return url


def getcolorim(ra, dec, size=240, output_size=None, filters="grizy", format="jpg"):
    """Get color image at a sky position

    ra, dec = position in degrees
    size = extracted image size in pixels (0.25 arcsec/pixel)
    output_size = output (display) image size in pixels (default = size).
                  output_size has no effect for fits format images.
    filters = string with filters to include
    format = data format (options are "jpg", "png")
    Returns the image
    """

    if format not in ("jpg", "png"):
        raise ValueError("format must be jpg or png")
    url = geturl(ra, dec, size=size, filters=filters, output_size=output_size, format=format, color=True)
    r = requests.get(url)
    im = Image.open(BytesIO(r.content))
    return im


def getgrayim(ra, dec, size=240, output_size=None, filter="g", format="jpg"):
    """Get grayscale image at a sky position

    ra, dec = position in degrees
    size = extracted image size in pixels (0.25 arcsec/pixel)
    output_size = output (display) image size in pixels (default = size).
                  output_size has no effect for fits format images.
    filter = string with filter to extract (one of grizy)
    format = data format (options are "jpg", "png")
    Returns the image
    """

    if format not in ("jpg", "png"):
        raise ValueError("format must be jpg or png")
    if filter not in list("grizy"):
        raise ValueError("filter must be one of grizy")
    url = geturl(ra, dec, size=size, filters=filter, output_size=output_size, format=format)
    r = requests.get(url[0])
    im = Image.open(BytesIO(r.content))
    return im


def plot_cutout(ra, dec, arcsec, interactive, title=None, fn=None, save=False, ax=None, **kwargs):
    arcsec_per_px = 0.25
    ang_px = int(arcsec / arcsec_per_px)
    ang_deg = arcsec / 3600

    plot_color_image = kwargs.get("plot_color_image", True)
    height = kwargs.pop('height', 2.5)
    imshow_kwargs = {
        'origin': 'upper',
        "extent": ([ra + ang_deg / 2, ra - ang_deg / 2, dec - ang_deg / 2, dec + ang_deg / 2])
    }
    scatter_args = [ra, dec]
    scatter_kwargs = {'marker': 'x', 'color': 'red'}

    if not plot_color_image:
        filters = 'grizy'
        if not ax:
            fig, axss = plt.subplots(2, len(filters), sharex='all', sharey='all',
                                     gridspec_kw={'wspace': 0, 'hspace': 0, 'height_ratios': [1, 8]},
                                     figsize=(height * 5, height))
        else:
            fig = plt.gcf()
            axss = ax

        for j, fil in enumerate(list(filters)):
            im = getgrayim(ra, dec, size=ang_px, filter=fil)
            axs = axss[1]
            axs[j].imshow(im, cmap='gray', **imshow_kwargs)

            axs[j].scatter(*scatter_args, **scatter_kwargs)
            axs[j].set_title(fil)
            axss[0][j].axis('off')

    else:
        logger.debug('plotting color image')
        if not ax:
            fig, axss = plt.subplots(figsize=(height, height))
        else:
            fig = plt.gcf()
            axss = ax

        im = getcolorim(ra, dec, size=ang_px)
        axss.imshow(im, **imshow_kwargs)
        axss.scatter(*scatter_args, **scatter_kwargs)

    _this_title = title if title else f"{ra}_{dec}"
    axss.set_title(_this_title)

    if save:
        logger.info(f'saving under {fn}')
        fig.savefig(fn)

    if interactive:
        return fig, axss

    plt.close()


def mastQuery(request):
    """Perform a MAST query.

    Parameters
    ----------
    request (dictionary): The MAST request json object

    Returns head,content where head is the response HTTP headers, and content is the returned data"""

    server = 'mast.stsci.edu'

    # Grab Python Version
    version = ".".join(map(str, sys.version_info[:3]))

    # Create Http Header Variables
    headers = {"Content-type": "application/x-www-form-urlencoded",
               "Accept": "text/plain",
               "User-agent": "python-requests/" + version}

    # Encoding the request as a json string
    requestString = json.dumps(request)
    requestString = urlencode(requestString)

    # opening the https connection
    conn = httplib.HTTPSConnection(server)

    # Making the query
    conn.request("POST", "/api/v0/invoke", "request=" + requestString, headers)

    # Getting the response
    resp = conn.getresponse()
    head = resp.getheaders()
    content = resp.read().decode('utf-8')

    # Close the https connection
    conn.close()

    return head, content


def resolve(name):
    """Get the RA and Dec for an object using the MAST name resolver

    Parameters
    ----------
    name (str): Name of object

    Returns RA, Dec tuple with position"""

    resolverRequest = {'service': 'Mast.Name.Lookup',
                       'params': {'input': name,
                                  'format': 'json'
                                  },
                       }
    headers, resolvedObjectString = mastQuery(resolverRequest)
    resolvedObject = json.loads(resolvedObjectString)
    # The resolver returns a variety of information about the resolved object,
    # however for our purposes all we need are the RA and Dec
    try:
        objRa = resolvedObject['resolvedCoordinate'][0]['ra']
        objDec = resolvedObject['resolvedCoordinate'][0]['decl']
    except IndexError as e:
        raise ValueError("Unknown object '{}'".format(name))
    return (objRa, objDec)


def fixcolnames(tab):
    """Fix column names returned by the casjobs query

    Parameters
    ----------
    tab (astropy.table.Table): Input table

    Returns reference to original table with column names modified"""

    pat = re.compile(r'\[(?P<name>[^[]+)\]')
    for c in tab.colnames:
        m = pat.match(c)
        if not m:
            raise ValueError("Unable to parse column name '{}'".format(c))
        newname = m.group('name')
        tab.rename_column(c, newname)
    return tab


def crossmatch_to_panstarrs(file, radius):
    """
    Crossmatch a catalogue to Pan-STARRS
    :param file: str or io stream, filename or io object of the CSV table
    :param radius: astropy.Quantity, search radius around catalogue sources
    :return: pandas.DataFrame
    """
    radius_arcsec = radius.to('degree').value
    logger.debug(f'searching in a radius of {radius}')

    if isinstance(file, str):
        file = open(file, 'rb')

    r = requests.post(crossmatch_url, params=dict(radius=radius_arcsec), files=dict(file=file))

    if r.status_code != 200:
        txt = r.text if r.text else r.reason
        raise ValueError(f'Crossmatch failed with status {r.status_code}: {txt}')

    restab = pd.read_csv(io.StringIO(r.text))
    uniuqe_ids = restab['_searchID_'].unique()
    logger.debug(f'found {len(restab)} results for {len(uniuqe_ids)} IDs')

    tab = pd.DataFrame(columns=restab.columns)
    for sid in tqdm.tqdm(uniuqe_ids, desc='selecting closest result'):
        m = restab['_searchID_'] == sid
        tab = tab.append(restab[m].iloc[restab['dstArcSec'][m].argmin()])

    logger.debug(f'final result has {len(tab)} rows')

    return tab


##################################
# END define some util functions #
##################################
