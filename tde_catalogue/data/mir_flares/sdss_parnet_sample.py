import os, argparse, logging, requests
import pandas as pd
from SciServer import CasJobs, Authentication

from tde_catalogue import main_logger, cache_dir, plots_dir
from tde_catalogue.data.mir_flares import base_name as mir_base_name
from tde_catalogue.utils.sdss_utils import get_sdss_credentials, plot_cutout, get_skyserver_token
from tde_catalogue.data.mir_flares.parent_sample import ParentSample


logger = main_logger.getChild(__name__)


class SDSSParentSample(ParentSample):

    base_name = f"{mir_base_name}/sdss_parent_sample"
    casjobs_table_name = "spectroscopic_galaxies"
    default_keymap = {
        'ra': 'ra',
        'dec': 'dec'
    }

    def __init__(self,
                 base_name=base_name,
                 store=True):

        uid, pw = get_sdss_credentials()
        logger.debug(f"logging in with {uid}")
        Authentication.login(uid, pw)

        self.base_name = base_name
        self._store = store

        # set up directories
        self.cache_dir = os.path.join(cache_dir, base_name)
        self.plots_dir = os.path.join(plots_dir, base_name)

        for d in [self.cache_dir, self.plots_dir]:
            if not os.path.isdir(d):
                os.makedirs(d)

        #######################################################################################
        # START make CASJOBS query #
        ############################

        if (not os.path.isfile(self.local_sample_copy)) or (not self._store):
            # If there is no local copy, get the table from CasJobs
            logger.info('No local copy of Panstarrs query result. Getting info from CasJobs')

            if not self._table_in_casjobs:
                # If the query result is not on CasJobs, do the query
                logger.info('Querying SDSS-CASJOBS')
                logger.debug(f'Query: {self.query}')
                self.job_id = jobId = CasJobs.submitJob(sql=self.query, context="DR16")
                logger.debug(f'Job {self.job_id}')
                v = logger.getEffectiveLevel() <= logging.DEBUG
                jobDescription = CasJobs.waitForJob(jobId=jobId, verbose=v)
                logger.debug(jobDescription["Message"])

            logger.debug('loading table from CASJOBS')
            self.df = CasJobs.getPandasDataFrameFromQuery(self._download_query, context='MyDB')
            logger.info(f'got {len(self.df)} objects')

            if self._store:
                logger.debug(f'saving to {self.local_sample_copy}')
                self.df.to_csv(self.local_sample_copy)

        ##########################
        # END make CASJOBS query #
        #######################################################################################

        if self._store:
            logger.info('loading local copy')
            self.df = pd.read_csv(self.local_sample_copy)

    @property
    def local_sample_copy(self):
        return os.path.join(self.cache_dir, 'sdss_query_result.csv')

    @property
    def _table_in_casjobs(self):
        tables_list = CasJobs.getTables()
        names = [e["Name"] for e in tables_list]
        return self.casjobs_table_name in names

    @property
    def query(self):
        q = f"""
        SELECT
            ra, dec, specObjID, bestObjID, fluxObjID, targetObjID, plateID, sciencePrimary
        FROM
            specObj
        INTO
            MyDB.{self.casjobs_table_name}
        WHERE
            class = 'GALAXY'
        """
        return q

    @property
    def _download_query(self):
        q = f"""
        SELECT *
        FROM {self.casjobs_table_name}
        """
        return q

    def plot_cutout(self, ind, **kwargs):
        ra, dec, id = float(self.df.ra[ind]), float(self.df.dec[ind]), str(self.df.bestObjID[ind])
        interactive = kwargs.get('interactive')
        if not interactive:
            kwargs['fn'] = os.path.join(self.plots_dir, f"{ind}_{id}.pdf")
            logger.info(f"saving under {kwargs['fn']}")
        return plot_cutout(ra, dec, **kwargs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--logging_level', type=str, default='INFO')
    cfg = parser.parse_args()

    main_logger.setLevel(cfg.logging_level)

    SDSSParentSample()