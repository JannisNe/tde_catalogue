import os, mastcasjobs, argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from tde_catalogue.utils.panstarrs_utils import plot_cutout
from tde_catalogue.data.mir_flares import base_name as mir_base_name

from timewise import ParentSampleBase
from timewise.general import main_logger

logger = main_logger.getChild(__name__)


class PanstarrsParentSample(ParentSampleBase):

    base_name = mir_base_name + '/panstarrs_parent_sample'
    MAST_table_name = 'test_table15_with_psc'
    ps_score_threshold = 0
    minDetections = 10

    default_keymap = {
        'dec': 'decMean',
        'ra': 'raMean',
        'id': 'objName'
    }

    def __init__(self,
                 base_name=base_name,
                 MAST_table_name=MAST_table_name,
                 ps_score_threshold=ps_score_threshold,
                 minDetections=minDetections,
                 store=True):

        super(PanstarrsParentSample, self).__init__(base_name=base_name)

        self.base_name = base_name
        self.MAST_table_name = MAST_table_name
        self.ps_score_threshold = ps_score_threshold
        self.minDetections = minDetections
        self._store = store

        # Set up mastcasjobs to query MAST
        self.mastcasjob = mastcasjobs.MastCasJobs(context="PanSTARRS_DR2")
        self.job_id = None

        #######################################################################################
        # START make MAST query #
        #########################

        if (not os.path.isfile(self.local_sample_copy)) or (not self._store):
            # If there is no local copy, get the table from MAST
            logger.info('No local copy of Panstarrs query result. Getting info from MAST')

            if not self.check_if_table_on_mast():
                # If the query result is not on MAST, do the query
                logger.info('Querying PANSTARRS-MAST')
                logger.debug(f'Query: {self.query}')
                self.job_id = self.mastcasjob.submit(self.query, task_name="parent sample query")
                logger.debug(f'Job {self.job_id}')
                self.mastcasjob.monitor(self.job_id)

            logger.debug('loading table from MAST')
            self.df = self.mastcasjob.get_table(self.MAST_table_name, format="CSV").to_pandas()
            logger.info(f'got {len(self.df)} objects')

            if self._store:
                logger.debug(f'saving to {self.local_sample_copy}')
                self.df.to_csv(self.local_sample_copy)

        #######################
        # END make MAST query #
        #######################################################################################

        if self._store:
            logger.info('loading local copy')
            self.df = pd.read_csv(self.local_sample_copy)

    @property
    def query(self):
        q = f"""
            SELECT
                o.objID, o.raMean, o.decMean, o.nDetections, o.objName
                psc.ps_score
            INTO
                MyDB.{self.MAST_table_name}
            FROM
                ObjectThin o 
                inner join HLSP_PS1_PSC.pointsource_scores psc 
                    on psc.objid=o.objid 
                        and psc.ps_score<={self.ps_score_threshold} 
                        and o.nDetections>={self.minDetections} 
            """
        return q

    def check_if_table_on_mast(self):
        """Checks whether the table is already in MyDB on MAST"""
        logger.debug(f'checking if {self.MAST_table_name} on MAST')
        q = 'SELECT Distinct TABLE_NAME FROM information_schema.TABLES'
        res = self.mastcasjob.quick(q, context='MyDB', task_name='listtables', system=True)
        # tables = self.mastcasjob.list_tables() This would be nice but errors.
        # Hopefully fixed by mastcasjobs team.
        table_exists = self.MAST_table_name in res['TABLE_NAME']
        if not table_exists:
            logger.debug(f'Table not on MAST!')
        return table_exists

    #####################################################################################################
    # START make some plotting functions #
    ######################################

    def plot_skymap(self):
        frac = 0.001
        sample = self.df.sample(frac=frac)
        fig = plt.figure()
        ax = fig.add_subplot(projection='mollweide')
        ax.scatter(np.deg2rad(sample.raMean - 360 / 2), np.deg2rad(sample.decMean), marker='.',
                   label=f'{frac}% of {len(self.df):.2e} sources')
        ax.grid()
        ax.legend()
        ax.set_title('PANSTARRS Sample Skymap')
        filename = os.path.join(self.plots_dir, 'sample_skymap.pdf')
        logger.info(f'saving under {filename}')
        fig.savefig(filename)
        plt.close()

    def _plot_cutout(self, ra, dec, arcsec, interactive, title=None, fn=None, **kwargs):
        _this_title = title if title else f"{ra}_{dec}"
        _this_fn = fn if fn else _this_title
        _filename = os.path.join(self.plots_dir, f'{_this_fn}.pdf')
        return plot_cutout(ra, dec, arcsec=arcsec, interactive=interactive,
                           fn=_filename, title=_this_title, **kwargs)

    ####################################
    # END make some plotting functions #
    ###################################################################################################


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--logging_level', type=str, default='INFO')
    cfg = parser.parse_args()

    main_logger.setLevel(cfg.logging_level)

    PanstarrsParentSample()