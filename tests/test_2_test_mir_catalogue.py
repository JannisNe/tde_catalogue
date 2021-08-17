import unittest, os, shutil, copy
import numpy as np
from astropy.coordinates import SkyCoord
from astropy import units as u

from tde_catalogue import main_logger, cache_dir, plots_dir
from tde_catalogue.utils.mirong_sample import get_mirong_sample
from tde_catalogue.data.mir_flares.panstarrs_parent_sample import PanstarrsParentSample
from tde_catalogue.data.mir_flares.wise_data import WISEData


main_logger.setLevel('DEBUG')
logger = main_logger.getChild(__name__)


mirong_sample = get_mirong_sample()
mirong_test_id = 28

test_ra = mirong_sample['RA'].iloc[mirong_test_id]
test_dec = mirong_sample['DEC'].iloc[mirong_test_id]
test_radius_arcsec = 3600


class PanstarrsParentSampleTestVersion(PanstarrsParentSample):
    """
    Does the same as PanstarrsParentSample but
    only on a confined region in the sky to deal with small tables
    """

    base_name = 'test/' + PanstarrsParentSample.base_name
    MAST_table_name = PanstarrsParentSample.MAST_table_name + '_test'
    query = f"""
    SELECT
        o.objID, o.raMean, o.decMean, o.nDetections, o.objName, o.objAltName1, o.objAltName2, o.objAltName3, o.objPopularName,
        psc.ps_score
    INTO
        MyDB.{MAST_table_name}
    FROM 
        fGetNearbyObjEq({test_ra},{test_dec},{test_radius_arcsec}/60.0) nb
        inner join ObjectThin o on o.objid=nb.objid and o.nDetections>10
        inner join HLSP_PS1_PSC.pointsource_scores psc on psc.objid=o.objid and psc.ps_score=0
    """

    def __init__(self):
        super().__init__(PanstarrsParentSampleTestVersion.base_name,
                         PanstarrsParentSampleTestVersion.MAST_table_name,
                         PanstarrsParentSampleTestVersion.query)

    def clean_up(self):
        logger.info(f'removing {cache_dir}')
        shutil.rmtree(self.cache_dir)
        logger.info(f'dropping {self.MAST_table_name} from MAST')
        self.mastcasjob.drop_table(self.MAST_table_name)


class WISEDataTestVersion(WISEData):
    """
    Same as WISEData but only for one confined region of the sky
    """
    base_name = 'test/' + WISEData.base_name

    def __init__(self):
        super().__init__(n_chunks=1,
                         base_name=WISEDataTestVersion.base_name,
                         parent_sample_class=PanstarrsParentSampleTestVersion)

    def clean_up(self):
        logger.info(f"removing {self.cache_dir}")
        shutil.rmtree(self.cache_dir)


class TestMIRFlareCatalogue(unittest.TestCase):

    def test_a_test_panstarrs_parent_sample(self):
        logger.info('\n\n Testing PanstarrsParentSample \n')
        logger.info('querying MAST and downloading table')
        pps = PanstarrsParentSampleTestVersion()
        logger.info('\n testing plots \n')
        pps.plot_skymap()
        pps.plot_cutout(0)

    def test_b_test_wise_data(self):
        logger.info('\n\n Testing WISE Data \n')
        wise_data = WISEDataTestVersion()
        wise_data.match_all_chunks()

        df = wise_data.parent_sample.df
        c1 = SkyCoord(df.raMean * u.degree, df.decMean * u.degree)
        c2 = SkyCoord(float(test_ra) * u.degree, float(test_dec) * u.degree)
        sep = c1.separation(c2)
        closest_ind = np.argsort(sep)

        self.assertLess(sep[closest_ind][0], 0.5 * u.arcsec)
        wise_data.parent_sample.plot_cutout(closest_ind[0], arcsec=40)

    @classmethod
    def tearDownClass(cls):
        logger.info('\n clean up \n')
        wise_data = WISEDataTestVersion()
        wise_data.clean_up()
        pps = PanstarrsParentSampleTestVersion()
        pps.clean_up()
