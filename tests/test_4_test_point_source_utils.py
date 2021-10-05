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
from tde_catalogue.utils.point_source_utils import get_point_source_wise_data


logger = main_logger.getChild(__name__)


test_ra = 243.2494163
test_dec = 42.3277313
base_name = 'test/' + WISEData.base_name + '_point_source_utils'


class TestPointSourceUtils(unittest.TestCase):

    def test_point_source_wise_data(self):
        logger.info("\n\n Testing Point Source Utils \n")
        wd = get_point_source_wise_data(base_name, test_ra, test_dec)
        wd.parent_sample.plot_cutout(parent_sample_idx='0')