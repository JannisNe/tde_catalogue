import unittest, os, shutil

from tde_catalogue import main_logger, cache_dir, plots_dir
from tde_catalogue.data.mir_flares.panstarrs_parent_sample import PanstarrsParentSample


logger = main_logger.getChild(__name__)

test_ra = 0
test_dec = 0
test_radius_arcsec = 3600


class TestPanstarrsParentSample(PanstarrsParentSample):
    """
    Does the same as PanstarrsParentSample but
    only on a confined region in the sky to deal with small tables
    """

    base_name = PanstarrsParentSample.base_name + '_test'
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
        super().__init__(TestPanstarrsParentSample.base_name,
                         TestPanstarrsParentSample.MAST_table_name,
                         TestPanstarrsParentSample.query)

    def clean_up(self):
        logger.info(f'removing {cache_dir}')
        shutil.rmtree(self.cache_dir)
        logger.info(f'dropping {self.MAST_table_name} from MAST')
        self.mastcasjob.drop_table(self.MAST_table_name)


class TestCompiler(unittest.TestCase):

    def test_a_query_and_download(self):
        logger.info('\n\n Testing PanstarrsParentSample \n')
        logger.info('querying MAST and downloading table')
        TestPanstarrsParentSample()

    def test_b_plots(self):
        logger.info('\n testing plots \n')
        pps = TestPanstarrsParentSample()
        pps.plot_skymap()
        pps.plot_cutout(0)

    @classmethod
    def tearDownClass(cls):
        logger.info('clean up')
        pps = TestPanstarrsParentSample()
        pps.clean_up()
