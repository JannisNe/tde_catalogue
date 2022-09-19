import os, argparse, logging
import pandas as pd
from SciServer import CasJobs, Authentication

from tde_catalogue import main_logger
from tde_catalogue.data.mir_flares import base_name as mir_base_name

from timewise import ParentSampleBase
from timewise.utils import get_sdss_credentials, plot_sdss_cutout as plot_cutout

logger = main_logger.getChild(__name__)


class SDSSParentSample(ParentSampleBase):

    base_name = f"{mir_base_name}/sdss_parent_sample"
    casjobs_table_name = "spectroscopic_galaxies"
    default_keymap = {
        'ra': 'ra',
        'dec': 'dec',
        'id': 'bestObjID'
    }

    def __init__(self,
                 base_name=base_name,
                 store=True,
                 submit_context='DR16',
                 download_context='MyDB'):

        super().__init__(base_name=base_name)

        uid, pw = get_sdss_credentials()
        logger.debug(f"logging in with {uid}")
        Authentication.login(uid, pw)

        self.base_name = base_name
        self._store = store
        self.submit_context = submit_context
        self.download_context = download_context

        #######################################################################################
        # START make CASJOBS query #
        ############################

        if (not os.path.isfile(self.local_sample_copy)) or (not self._store):
            # If there is no local copy, get the table from CasJobs
            logger.info('No local copy of SDSS query result. Getting info from CasJobs')

            if not self._table_in_casjobs:
                # If the query result is not on CasJobs, do the query
                logger.info('Querying SDSS-CASJOBS')
                logger.debug(f'Query: {self.query}')
                self.job_id = jobId = CasJobs.submitJob(sql=self.query, context=self.submit_context)
                logger.debug(f'Job {self.job_id}')
                v = logger.getEffectiveLevel() <= logging.DEBUG
                jobDescription = CasJobs.waitForJob(jobId=jobId, verbose=v)
                logger.debug(jobDescription["Message"])

            logger.debug('loading table from CASJOBS')
            self.df = CasJobs.getPandasDataFrameFromQuery(self._download_query, context=self.download_context)
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
    def _table_in_casjobs(self):
        tables_list = CasJobs.getTables()
        names = [e["Name"] for e in tables_list]
        return self.casjobs_table_name in names

    @property
    def query(self):
        q = f"""
        SELECT
            s.ra, s.dec, p.u, p.g, p.r, p.i, p.z, s.z as redshift
        FROM
            specObj as s
            INNER JOIN photoObj p on p.objID = s.bestObjID
        INTO
            {self.download_context}.{self.casjobs_table_name}
        WHERE
            s.class = 'GALAXY' OR s.class = 'QSO'
        """
        return q

    @property
    def _download_query(self):
        q = f"""
        SELECT *
        FROM {self.casjobs_table_name}
        """
        return q

    def _plot_cutout(self, ra, dec, arcsec, interactive, **kwargs):
        return plot_cutout(ra, dec, arcsec=arcsec, interactive=interactive, **kwargs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--logging_level', type=str, default='INFO')
    cfg = parser.parse_args()

    main_logger.setLevel(cfg.logging_level)

    SDSSParentSample()