import getpass, os
from SciServer import SkyServer
import matplotlib.pyplot as plt


def get_sdss_credentials():
    if not os.environ.get('SDSS_USERID'):
        os.environ['SDSS_USERID'] = input('Enter SDSS user ID:')
    if not os.environ.get('SDSS_USERPW'):
        os.environ['SDSS_USERPW'] = getpass.getpass('Enter SDSS password:')
    return os.environ['SDSS_USERID'], os.environ['SDSS_USERPW']


def get_skyserver_token():
    return os.getenv("SKYSERVER_TOKEN")


def plot_cutout(ra, dec, arcsec=20, arcsec_per_px=1, interactive=False, fn=None):
    ang_px = int(arcsec / arcsec_per_px)
    ang_deg = arcsec / 3600

    im = SkyServer.getJpegImgCutout(ra, dec, scale=arcsec_per_px, height=ang_px, width=ang_px)

    fig, ax = plt.subplots()
    ax.imshow(im, origin='upper',
              extent=([ra + ang_deg / 2, ra - ang_deg / 2,
                       dec - ang_deg / 2, dec + ang_deg / 2]),
              cmap='gray')

    ax.scatter(ra, dec, marker='x', color='red')

    if interactive:
        return fig, ax
    else:
        fig.savefig(fn)
        plt.close()