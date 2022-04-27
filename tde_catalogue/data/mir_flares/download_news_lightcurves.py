import logging
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
        n_chunks=80,
        min_sep_arcsec=6
    )

    wd.get_sample_photometric_data(
        max_nTAPjobs=4,
        perc=1,
        tables=None,#"NEOWISE-R Single Exposure (L1b) Source Table",
        chunks=None,#list(range(38, 40)),
        cluster_jobs_per_chunk=50,
        wait=3,
        remove_chunks=False,
        query_type='by_allwise_id',
        overwrite=True
    )