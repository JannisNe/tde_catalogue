import os, unittest, argparse
from astropy.coordinates import SkyCoord
from astropy import units as u

from tde_catalogue import main_logger
from tde_catalogue.utils.mirong_sample import get_mirong_sample
from tde_catalogue.data.mir_flares import base_name as mir_base_name
from tde_catalogue.data.mir_flares.combined_parent_sample import CombinedParentSample

from timewise import WiseDataByVisit

logger = main_logger.getChild(__name__)
this_base_name = f'test/{mir_base_name}/WISEData_mirong_sources'


class TestMIRONGFlaresParentSample(CombinedParentSample):

    base_name = CombinedParentSample.base_name + this_base_name

    def __init__(self, radius_around_mirong_sources_arcmin=2, base_name=base_name, **kwargs):
        self._r = radius_around_mirong_sources_arcmin * u.arcmin
        super().__init__(base_name=base_name, **kwargs)

    def _combine_samples(self):
        # temporary disable storing
        _store = self._store
        self._store = False
        super(TestMIRONGFlaresParentSample, self)._combine_samples()
        # re enable storing
        self._store = _store

        logger.info('selecting sources around MIRONG sources')
        mirong_sample = get_mirong_sample()
        mirong_coords = SkyCoord(mirong_sample.RA, mirong_sample.DEC, unit='deg')
        my_coords = SkyCoord(self.df.ra, self.df.dec, unit='deg')
        my_idx, mirong_idx, d2d, d3d = mirong_coords.search_around_sky(my_coords, self._r)
        logger.debug(f"found {len(my_idx)} sources (out of {len(self.df)})")
        self.df = self.df.iloc[my_idx]
        if self._store:
            self.save_local()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--logging_level', type=str, default='INFO', nargs='?', const='DEBUG')
    cfg = parser.parse_args()
    main_logger.setLevel(cfg.logging_level)

    wise_data = WiseDataByVisit(base_name=this_base_name,
                                parent_sample_class=TestMIRONGFlaresParentSample,
                                n_chunks=15,
                                min_sep_arcsec=8)
    wise_data.match_all_chunks()
    # wise_data.get_photometric_data(mag=True, flux=True, service='gator')
    wise_data._select_and_bin_lightcurves_gator()
    wise_data._combine_binned_lcs('gator')