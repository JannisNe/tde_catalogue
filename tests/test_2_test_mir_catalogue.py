import unittest, os, shutil, copy
import numpy as np

from tde_catalogue import main_logger, cache_dir, plots_dir
from tde_catalogue.utils.mirong_sample import get_mirong_sample
from tde_catalogue.data.mir_flares.panstarrs_parent_sample import PanstarrsParentSample
from tde_catalogue.data.mir_flares.wise_data import WISEData


main_logger.setLevel('DEBUG')
logger = main_logger.getChild(__name__)


mirong_sample = get_mirong_sample()
mirong_test_id = 0

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
    where = f"""
        WHERE
            CONTAINS(POINT('ICRS',ra, dec), CIRCLE('ICRS',{test_ra},{test_dec},{test_radius_arcsec/3600}))=1"""

    def __init__(self):
        super().__init__(n_chunks=3,
                         where=WISEDataTestVersion.where,
                         base_name=WISEDataTestVersion.base_name,
                         parent_sample=PanstarrsParentSampleTestVersion)
        self._cached_tap_output = None

    def get_tap_output(self, chunk_number, table_name):
        dec_intervall = self.dec_intervalls[chunk_number]
        logger.info(f'getting TAP output for DEC interval {dec_intervall}')

        if isinstance(self._cached_tap_output, type(None)):
            logger.debug('No cached TAP output')

            queue = f"""
            SELECT
                source_id, ra, dec, sigra, sigdec, nb, na, cc_flags
            FROM
                {table_name}
            WHERE
                CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS',{test_ra},{test_dec},{test_radius_arcsec/3600}))
            """

            logger.info(f"Queue: {queue}")
            query_job = WISEData.service.submit_job(queue)
            query_job.run()
            logger.info(f'Job: {query_job.url}; {query_job.phase}')
            logger.info('waiting ...')
            query_job.wait()
            logger.info('Done!')

            self._cached_tap_output = query_job.fetch_result().to_table().to_pandas()

        m = (self._cached_tap_output.nb < 2) & (self._cached_tap_output.na < 1)
        dec_m = (self._cached_tap_output.dec > min(dec_intervall)) & (self._cached_tap_output.dec < max(dec_intervall))
        cc_m = np.array([cc.startswith('00') for cc in self._cached_tap_output.cc_flags])
        tap_res = copy.copy(self._cached_tap_output[m & cc_m & dec_m])
        logger.debug(f'Found {len(tap_res)} objects.')

        return tap_res


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
        wise_data.match_all_chunks('allwise_p3as_psd')

    @classmethod
    def tearDownClass(cls):
        logger.info('\n clean up \n')
        pps = PanstarrsParentSampleTestVersion()
        pps.clean_up()
