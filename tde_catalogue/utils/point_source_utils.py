import pandas as pd
import matplotlib.pyplot as plt

from tde_catalogue import main_logger
from timewise import ParentSampleBase, WiseDataByVisit
from timewise.utils import plot_sdss_cutout as sdss_cutout
from tde_catalogue.utils.panstarrs_utils import plot_cutout as panstarrs_cutout


logger = main_logger.getChild(__name__)


def get_point_source_parent_sample(base_name, ra, dec):

    class PointSourceParentSample(ParentSampleBase):
        default_keymap = {
            'ra': 'ra',
            'dec': 'dec',
            'id': 'id'
        }

        def __init__(self):

            super().__init__(base_name=base_name)

            self.base_name = base_name
            self.df = pd.DataFrame({'ra': [ra], 'dec': [dec], 'id': ['NGC 1068']})

        def _plot_cutout(self, ra, dec, arcsec, interactive, title=None, fn=None, save=False, ax=None, **kwargs):
            h = kwargs.get('height', 2)
            if not ax:
                fig, ax = plt.subplots(ncols=2, figsize=(h * 2, h), sharex='all')
            else:
                fig = plt.gcf()
            sdss_cutout(ra, dec, interactive=True, ax=ax[0], arcsec=arcsec, title='SDSS', **kwargs)
            panstarrs_cutout(ra, dec, interactive=True, ax=ax[1], arcsec=arcsec, title='PanSTARRS', **kwargs)

            if interactive:
                return fig, ax
            if save:
                fig.savefig(fn)
                plt.close()

            plt.show()
            plt.close()

        def save_local(self):
            logger.debug(f"not saving")

    return PointSourceParentSample


def get_point_source_wise_data(base_name, ra, dec, min_sep_arcsec=10, **kwargs):
    ps = get_point_source_parent_sample(base_name, ra, dec)
    wd = WiseDataByVisit(n_chunks=1, base_name=base_name, parent_sample_class=ps, min_sep_arcsec=min_sep_arcsec)
    wd.match_all_chunks()
    wd.get_photometric_data(**kwargs)
    wd.plot_lc(parent_sample_idx=0, service=kwargs.get('service', 'tap'))
    return wd