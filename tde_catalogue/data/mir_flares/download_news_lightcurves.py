import os, logging
import numpy as np

from timewise import ParentSampleBase, WiseDataByVisit, WISEDataDESYCluster
from timewise.general import data_dir, main_logger

from tde_catalogue.data.mir_flares.news_parent_sample import NEWSParentSample
from tde_catalogue.data.mir_flares.news_wise_data import WISEDataWithKnownDesignation


logger = logging.getLogger(__name__)


if __name__ == '__main__':
    logging.getLogger().setLevel('DEBUG')
    logger.setLevel('DEBUG')
    logger.info('initialising wise data')
    wd = WISEDataWithKnownDesignation(
        base_name=NEWSParentSample.base_name,
        parent_sample_class=NEWSParentSample,
        n_chunks=90,
        min_sep_arcsec=6
    )

    wd.clear_unbinned_photometry_when_binning = False

    wd.get_sample_photometric_data(
        max_nTAPjobs=4,
        perc=1,
        tables=None,
        chunks=list(range(10)),
        cluster_jobs_per_chunk=100,
        wait=5,
        remove_chunks=False,
        query_type='by_allwise_id',
        overwrite=True
    )