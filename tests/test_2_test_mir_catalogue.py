import unittest, shutil, argparse, time, os
import numpy as np
from astropy.coordinates import SkyCoord
from astropy import units as u

from tde_catalogue import main_logger, cache_dir
from tde_catalogue.utils.mirong_sample import get_mirong_sample
from tde_catalogue.data.mir_flares.panstarrs_parent_sample import PanstarrsParentSample
from tde_catalogue.data.mir_flares.wise_data import WISEData
from tde_catalogue.data.mir_flares.sdss_parnet_sample import SDSSParentSample, CasJobs
from tde_catalogue.data.mir_flares.combined_parent_sample import CombinedParentSample


main_logger.setLevel('DEBUG')
logger = main_logger.getChild(__name__)


mirong_sample = get_mirong_sample()
mirong_test_id = 28

test_ra = mirong_sample['RA'].iloc[mirong_test_id]
test_dec = mirong_sample['DEC'].iloc[mirong_test_id]
test_radius_arcsec = 3600


###########################################################################################################
# START DEFINING TEST CLASSES      #
####################################


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
        # logger.info(f'dropping {self.MAST_table_name} from MAST')
        # self.mastcasjob.drop_table(self.MAST_table_name)


class SDSSParentSampleTestVersion(SDSSParentSample):
    base_name = 'test/' + SDSSParentSample.base_name
    casjobs_table_name = SDSSParentSample.casjobs_table_name + '_test'

    def __init__(self):
        super().__init__(self.base_name)

    def clean_up(self):
        logger.info(f"removing {self.cache_dir}")
        shutil.rmtree(self.cache_dir)
        logger.info(f"dropping {self.casjobs_table_name} from CasJobs")
        df = CasJobs.executeQuery(sql="DROP TABLE " + self.casjobs_table_name, context="MyDB", format="pandas")
        logger.debug(df)

    @property
    def query(self):
        q = f"""
        SELECT
            o.ra, o.dec, o.specObjID, o.bestObjID, o.fluxObjID, o.targetObjID, o.plateID, o.sciencePrimary
        FROM
            fGetNearbyObjEq({test_ra},{test_dec},{test_radius_arcsec}/60.0) nb
            inner join specObj o on nb.objID = o.bestObjID
        INTO
            MyDB.{self.casjobs_table_name}
        WHERE
            class = 'GALAXY'
        """
        return q


class CombinedSampleTestVersion(CombinedParentSample):
    base_name = 'test/' + CombinedParentSample.base_name

    def __init__(self):
        super().__init__([SDSSParentSampleTestVersion, PanstarrsParentSampleTestVersion],
                         base_name=CombinedSampleTestVersion.base_name)

    def clean_up(self):
        logger.info(f"removing {self.cache_dir}")
        shutil.rmtree(self.cache_dir)


class WISEDataTestVersion(WISEData):
    """
    Same as WISEData but only for one confined region of the sky
    """
    base_name = 'test/' + WISEData.base_name

    def __init__(self, name_ext=''):
        super().__init__(n_chunks=10,
                         base_name=WISEDataTestVersion.base_name + name_ext,
                         parent_sample_class=CombinedSampleTestVersion)

    def get_photometric_data(self, tables=None, perc=1, wait=0, service='tap', mag=True, flux=True,
                             nthreads=100, chunks=None, cluster_jobs_per_chunk=0,
                             overwrite=True, remove_chunks=True):
        if tables is None:
            tables = ['AllWISE Multiepoch Photometry Table']
        super(WISEDataTestVersion, self).get_photometric_data(tables, perc, wait, service, mag, flux)
        
    def clean_up(self):
        logger.info(f"removing {self.cache_dir}")
        shutil.rmtree(self.cache_dir)


####################################
# END DEFINING TEST CLASSES        #
###########################################################################################################


class TestMIRFlareCatalogue(unittest.TestCase):

    def test_a_test_panstarrs_parent_sample(self):
        logger.info('\n\n Testing PanstarrsParentSample \n')
        logger.info('querying MAST and downloading table')
        pps = PanstarrsParentSampleTestVersion()
        logger.info('\n testing plots \n')
        pps.plot_skymap()
        pps.plot_cutout(0)

    def test_b_test_sdss_parent_sample(self):
        logger.info("\n\n Testing SDSS Parent Sample\n")
        logger.info("query CasJobs")
        SDSSParentSampleTestVersion()

    def test_c_test_combined_sample(self):
        logger.info("\n\n Testing Combined Parent Sample\n")
        CombinedSampleTestVersion()

    def test_d_test_wise_data(self):
        logger.info('\n\n Testing WISE Data \n')
        wise_data = WISEDataTestVersion()
        wise_data.match_all_chunks()

        df = wise_data.parent_sample.df
        c1 = SkyCoord(df.ra * u.degree, df.dec * u.degree)
        c2 = SkyCoord(float(test_ra) * u.degree, float(test_dec) * u.degree)
        sep = c1.separation(c2)
        closest_ind = np.argsort(sep)

        self.assertLess(sep[closest_ind][0], 0.5 * u.arcsec)
        wise_data.parent_sample.plot_cutout(closest_ind[0], arcsec=40)

        logger.info(f"\n\n Testing getting photometry \n")
        for s in ['gator', 'tap']:
            logger.info(f"\nTesting {s.upper()}")
            wise_data.get_photometric_data(service=s, mag=True, flux=True)
            logger.info(f" --- Test plot lightcurves --- ")
            lcs = wise_data.load_binned_lcs(s)
            plot_id = list(lcs.keys())[10].split('_')[0]
            for lumk in ['mag', 'flux']:
                fn = os.path.join(wise_data.plots_dir, f"{plot_id}.pdf")
                wise_data.plot_lc(parent_sample_idx=plot_id, plot_unbinned=True, lum_key=lumk, service=s, fn=fn)

    @classmethod
    def tearDownClass(cls):
        logger.info('\n clean up \n')
        wise_data = WISEDataTestVersion()
        wise_data.clean_up()
        pps = PanstarrsParentSampleTestVersion()
        pps.clean_up()
        sdss_test = SDSSParentSampleTestVersion()
        sdss_test.clean_up()
        combined_sample = CombinedSampleTestVersion()
        combined_sample.clean_up()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--logging_level', type=str, default='INFO', const='DEBUG', nargs='?')
    parser.add_argument('-p', '--percent', type=float, default=1)
    parser.add_argument('--service', type=str, default='tap')
    parser.add_argument('-fn', '--filename', type=str, default='')
    cfg = parser.parse_args()

    main_logger.setLevel(cfg.logging_level)

    start_time = time.time()
    name_ext = '' if cfg.percent == 1 else f"{cfg.percent*100:.0f}percent_of_sources"
    wise_data = WISEDataTestVersion(name_ext=name_ext)
    init_time = time.time()
    wise_data.match_all_chunks()
    match_time = time.time()
    wise_data.get_photometric_data(perc=cfg.percent, service=cfg.service)
    tables = ['AllWISE Multiepoch Photometry Table']
    phot_time = time.time()

    txt = f"{cfg.percent*100}% of {len(wise_data.parent_sample.df)} sources:\n" \
          f"Total Time: {phot_time - start_time} \n" \
          f"\tInitialising:\t{init_time - start_time} \n" \
          f"\tMatching:\t{match_time - init_time} \n" \
          f"\tPhotometry\t{phot_time - match_time}"
    logger.info(txt)

    if cfg.filename:
        logger.info(f'writing to {cfg.filename}')
        with open(cfg.filename, 'w') as f:
            f.write(txt)