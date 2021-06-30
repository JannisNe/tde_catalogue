import os
import mastcasjobs
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from tde_catalogue import main_logger
from tde_catalogue import cache_dir, plots_dir
from tde_catalogue.utils.panstarrs_utils import getgrayim
from tde_catalogue.data.mir_flares import base_name as mir_base_name


logger = main_logger.getChild(__name__)


class PanstarrsParentSample:

    base_name = mir_base_name + '/panstarrs_parent_sample'
    MAST_table_name = 'test_table15_with_psc'

    query = f"""
    SELECT
        o.objID, o.raMean, o.decMean, o.nDetections, o.objName, o.objAltName1, o.objAltName2, o.objAltName3, o.objPopularName,
        psc.ps_score
    INTO
        MyDB.{MAST_table_name}
    FROM
        ObjectThin o 
        inner join HLSP_PS1_PSC.pointsource_scores psc on psc.objid=o.objid and psc.ps_score=0 and o.nDetections>10 
    """

    def __init__(self,
                 base_name=base_name,
                 MAST_table_name=MAST_table_name,
                 query=query):

        self.base_name = base_name
        self.MAST_table_name = MAST_table_name
        self.query = query

        # Set up mastcasjobs to query MAST
        self.mastcasjob = mastcasjobs.MastCasJobs(context="PanSTARRS_DR2")
        self.job_id = None

        # set up directories
        self.cache_dir = os.path.join(cache_dir, base_name)
        self.plots_dir = os.path.join(plots_dir, base_name)

        for d in [self.cache_dir, self.plots_dir]:
            if not os.path.isdir(d):
                os.makedirs(d)

        #######################################################################################
        # START make MAST query #
        #########################

        if not os.path.isfile(self.local_panstarrs_sample_copy):
            # If there is no local copy, get the table from MAST
            logger.info('No local copy of Panstarrs query result. Getting info from MAST')

            if not self.check_if_table_on_mast():
                # If the query result is not on MAST, do the query
                logger.info('Querying PANSTARRS-MAST')
                logger.info(f'Query: {self.query}')
                self.job_id = self.mastcasjob.submit(self.query, task_name="parent sample query")
                logger.info(f'Job {self.job_id}')
                self.mastcasjob.monitor(self.job_id)

            logger.info('loading table from MAST')
            results = self.mastcasjob.get_table(self.MAST_table_name, format="CSV")
            logger.info(f'got {len(results)} objects')
            logger.info(f'saving to {self.local_panstarrs_sample_copy}')
            results.to_pandas().to_csv(self.local_panstarrs_sample_copy)

        #######################
        # END make MAST query #
        #######################################################################################

        logger.info('loading local copy')
        self.df = pd.read_csv(self.local_panstarrs_sample_copy)

    @property
    def local_panstarrs_sample_copy(self):
        return os.path.join(self.cache_dir, 'panstarrs_query_result.csv')

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

    ######################################
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

    def plot_cutout(self, ind, arcsec=20):
        """
        Plot the coutout images in all filters around the position of object with index i
        """

        for i in np.atleast_1d(ind):

            r = self.df.iloc[i]
            arcsec_per_px = 0.25
            ang_px = int(arcsec / arcsec_per_px)
            ang_deg = arcsec / 3600

            filters = 'grizy'
            height = 2.5
            fig, axss = plt.subplots(2, len(filters), sharex='all', sharey='all',
                                     gridspec_kw={'wspace': 0, 'hspace': 0, 'height_ratios': [1, 8]},
                                     figsize=(height * 5, height))
            for j, fil in enumerate(list(filters)):
                im = getgrayim(r.raMean, r.decMean, size=ang_px, filter=fil)
                axs = axss[1]
                axs[j].imshow(im, origin='upper',
                              extent=([r.raMean - ang_deg, r.raMean + ang_deg, r.decMean - ang_deg, r.decMean + ang_deg]),
                              cmap='gray')

                axs[j].scatter(r.raMean, r.decMean, marker='x', color='red')
                axs[j].set_title(fil)
                axss[0][j].axis('off')

            fig.suptitle(r.objName)

            filename = os.path.join(self.plots_dir, f'{i}_{r.objName}.pdf')
            logger.info(f'saving under {filename}')
            fig.savefig(filename)
            plt.close()

    ####################################
    # END make some plotting functions #
    ####################################
