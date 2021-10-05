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


def plot_cutout(ra, dec, arcsec=20, arcsec_per_px=0.1, interactive=False, fn=None, title=None, save=False, ax=False,
                height=2.5):

    ang_px = int(arcsec / arcsec_per_px)
    ang_deg = arcsec / 3600

    if not ax:
        fig, ax = plt.subplots(figsize=(height, height))
    else:
        fig = plt.gcf()

    try:
        im = SkyServer.getJpegImgCutout(ra, dec, scale=arcsec_per_px, height=ang_px, width=ang_px)
        ax.imshow(im, origin='upper',
                  extent=([ra + ang_deg / 2, ra - ang_deg / 2,
                           dec - ang_deg / 2, dec + ang_deg / 2]),
                  cmap='gray')
        ax.scatter(ra, dec, marker='x', color='red')

    except Exception as e:
        if "outside the SDSS footprint" in str(e):
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            x = sum(xlim) / 2
            y = sum(ylim) / 2
            ax.annotate("Outside SDSS Footprint", (x, y), color='red', ha='center', va='center', fontsize=20)
        else:
            raise Exception(e)

    if title:
        ax.set_title(title)

    if save:
        fig.savefig(fn)

    if interactive:
        return fig, ax

    plt.close()