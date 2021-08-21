import os, subprocess, copy, json, argparse
import pandas as pd
import numpy as np
import pyvo as vo
import astropy.units as u
from astropy.table import Table

from tde_catalogue import main_logger, cache_dir, plots_dir, output_dir
from tde_catalogue.data.mir_flares import base_name as mir_base_name
from tde_catalogue.data.mir_flares.panstarrs_parent_sample import PanstarrsParentSample

logger = main_logger.getChild(__name__)


class WISEData:

    base_name = mir_base_name + '/WISE_data'
    service_url = 'https://irsa.ipac.caltech.edu/TAP'
    service = vo.dal.TAPService(service_url)

    table_names = pd.DataFrame([
        ('AllWISE Multiepoch Photometry Table', 'allwise_p3as_mep'),
        ('AllWISE Source Catalog', 'allwise_p3as_psd'),
        ('WISE 3-Band Cryo Single Exposure (L1b) Source Table', 'allsky_3band_p1bs_psd'),
        ('NEOWISE-R Single Exposure (L1b) Source Table', 'neowiser_p1bs_psd'),

    ], columns=['nice_table_name', 'table_name'])

    photometry_table_keymap = {
        'AllWISE Multiepoch Photometry Table': {
            'w1flux_ep':    'W1_flux',
            'w1sigflux_ep': 'W1_flux_error',
            'w2flux_ep':    'W2_flux',
            'w2sigflux_ep': 'W2_flux_error'
        },
        'NEOWISE-R Single Exposure (L1b) Source Table': {
            'w1flux':       'W1_flux',
            'w1sigflux':    'W1_flux_error',
            'w2flux':       'W2_flux',
            'w2sigflux':    'W2_flux_error'
        }
    }

    constraints = [
        "nb < 2",
        "na < 1",
        "cc_flags like '00%'",
        "qi_fact >= 1",
        "saa_sep >= 5",
        "moon_masked = '00'"
    ]

    bands = ['W1', 'W2']
    flux_key_ext = "_flux"
    error_key_ext = "_flux_error"
    band_plot_colors = {'W1': 'r', 'W2': 'b'}

    def __init__(self,
                 min_sep_arcsec=60,
                 n_chunks=8,
                 base_name=base_name,
                 parent_sample_class=PanstarrsParentSample):
        """
        Initialise a class instance
        :param min_sep_arcsec: float, minimum separation required to the parent sample sources
        :param n_chunks: int, number of chunks in declination
        :param base_name: str, unique name to determine storage directories
        :param parent_sample_class: object, class for parent sample
        """

        #######################################################################################
        # START SET-UP          #
        #########################

        parent_sample = parent_sample_class()
        self.base_name = base_name
        self.min_sep = min_sep_arcsec * u.arcsec

        # set up parent sample keys
        self.parent_ra_key = parent_sample.default_keymap['ra']
        self.parent_dec_key = parent_sample.default_keymap['dec']
        self.parent_wise_source_id_key = 'WISE_id'
        self.parent_sample_wise_skysep_key = 'sep_to_WISE_source'

        # set up directories
        self.cache_dir = os.path.join(cache_dir, base_name)
        self.output_dir = os.path.join(output_dir, base_name)
        self.lightcurve_dir = os.path.join(self.output_dir, "lightcurves")
        self.plots_dir = os.path.join(plots_dir, base_name)

        for d in [self.cache_dir, self.output_dir, self.lightcurve_dir, self.plots_dir]:
            if not os.path.isdir(d):
                os.makedirs(d)

        # set up result attributes
        self._all_lightcurves = None
        self.binned_lightcurves = None
        self._cached_binned_lcs_json = os.path.join(self.cache_dir, "binned_lcs.json")

        #########################
        # END SET-UP            #
        #######################################################################################

        self.parent_sample = parent_sample
        min_dec = np.floor(min(self.parent_sample.df[self.parent_dec_key]))
        max_dec = np.ceil(max(self.parent_sample.df[self.parent_dec_key]))
        logger.info(f'Declination: ({min_dec}, {max_dec})')
        self.parent_sample.df[self.parent_wise_source_id_key] = ""
        self.parent_sample.df[self.parent_sample_wise_skysep_key] = np.inf

        sin_bounds = np.linspace(np.sin(np.radians(min_dec)), np.sin(np.radians(max_dec)), n_chunks+1, endpoint=True)
        self.dec_intervalls = np.degrees(np.arcsin(np.array([sin_bounds[:-1], sin_bounds[1:]]).T))
        logger.info(f'Declination intervalls are {self.dec_intervalls}')

    @staticmethod
    def get_db_name(table_name, nice=False):
        """
        Get the right table name
        :param table_name: str, table name
        :param nice: bool, whether to get the nice table name
        :return: str
        """
        source_column = 'nice_table_name' if not nice else 'table_name'
        target_column = 'table_name' if not nice else 'nice_table_name'

        m = WISEData.table_names[source_column] == table_name
        if np.any(m):
            table_name = WISEData.table_names[target_column][m].iloc[0]
        else:
            logger.debug(f"{table_name} not in Table. Assuming it is the right name already.")
        return table_name

    ###########################################################################################################
    # START MATCH PARENT SAMPLE TO WISE SOURCES         #
    #####################################################

    def match_all_chunks(self, **table_name):
        for i in range(len(self.dec_intervalls)):
            self.match_single_chunk(i, **table_name)

    def match_single_chunk(self, chunk_number,
                           table_name="AllWISE Source Catalog"):
        """
        Match the parent sample to WISE
        :param chunk_number: int, number of the declination chunk
        :param table_name: str, optional, WISE table to match to, default is AllWISE Source Catalog
        """

        # select the parent sample in this declination range
        dec_intervall = self.dec_intervalls[chunk_number]
        logger.debug(f"interval is {dec_intervall}")
        dec_intervall_mask = (self.parent_sample.df[self.parent_dec_key] > min(dec_intervall)) & \
                             (self.parent_sample.df[self.parent_dec_key] < max(dec_intervall))
        logger.debug(f"Any selected: {np.any(dec_intervall_mask)}")

        selected_parent_sample = copy.copy(self.parent_sample.df.loc[dec_intervall_mask,
                                                                     [self.parent_ra_key, self.parent_dec_key]])
        selected_parent_sample.rename(columns={self.parent_dec_key: 'dec',
                                               self.parent_ra_key: 'ra'},
                                      inplace=True)
        logger.debug(f"{len(selected_parent_sample)} in dec interval")

        # write to IPAC formatted table
        _selected_parent_sample_astrotab = Table.from_pandas(selected_parent_sample)
        _parent_sample_declination_band_file = os.path.join(self.cache_dir, f"parent_sample_chunk{chunk_number}.xml")
        logger.debug(f"writing {len(_selected_parent_sample_astrotab)} "
                     f"objects to {_parent_sample_declination_band_file}")
        _selected_parent_sample_astrotab.write(_parent_sample_declination_band_file, format='ipac', overwrite=True)

        # use Gator to query IRSA
        _output_file = os.path.join(self.cache_dir, f"wise_data_chunk{chunk_number}.tbl")
        submit_cmd = f'curl ' \
                     f'-o {_output_file} ' \
                     f'-F filename=@{_parent_sample_declination_band_file} ' \
                     f'-F catalog={self.get_db_name(table_name)} ' \
                     f'-F spatial=Upload ' \
                     f'-F uradius={self.min_sep.to("arcsec").value} ' \
                     f'-F outfmt=1 ' \
                     f'-F one_to_one=1 ' \
                     f'-F selcols=designation,source_id,ra,dec,sigra,sigdec,cntr ' \
                     f'"https://irsa.ipac.caltech.edu/cgi-bin/Gator/nph-query"'

        logger.debug(f'submit command: {submit_cmd}')
        process = subprocess.Popen(submit_cmd, stdout=subprocess.PIPE, shell=True)
        out_msg, err_msg = process.communicate()
        logger.info(out_msg.decode())
        if err_msg:
            logger.error(err_msg.decode())

        # load the result file
        gator_res = Table.read(_output_file, format='ipac')
        logger.debug(f"found {len(gator_res)} results")

        # insert the corresponding separation to the WISE source into the parent sample
        self.parent_sample.df.loc[
            dec_intervall_mask,
            self.parent_sample_wise_skysep_key
        ] = list(gator_res["dist_x"])

        # insert the corresponding WISE IDs into the parent sample
        self.parent_sample.df.loc[
            dec_intervall_mask,
            self.parent_wise_source_id_key
        ] = list(gator_res["cntr"])

    ###################################################
    # END MATCH PARENT SAMPLE TO WISE SOURCES         #
    ###########################################################################################################

    ###########################################################################################################
    # START GET PHOTOMETRY DATA       #
    ###################################

    def get_photometric_data(self, tables=None):
        """
        Load photometric data from the IRSA server for the matched sample
        :param tables: list like, WISE tables to use for photometry query, defaults to AllWISE and NOEWISER photometry
        :return: pandas.DataFrame
        """

        if tables is None:
            tables = [
                'NEOWISE-R Single Exposure (L1b) Source Table',
                'AllWISE Multiepoch Photometry Table'
            ]

        self._query_for_photometry(tables)
        self._select_individual_lightcurves_and_bin()
        self._cache_binned_lightcurves()

    def _get_photometry_query_string(self, table_name):
        """
        Construct a query string to submit to IRSA
        :param table_name: str, table name
        :return: str
        """
        logger.debug(f"constructing query for {table_name}")
        db_name = self.get_db_name(table_name)
        nice_name = self.get_db_name(table_name, nice=True)
        flux_keys = list(self.photometry_table_keymap[nice_name].keys())
        keys = ['mjd'] + flux_keys
        id_key = 'cntr_mf' if 'allwise' in db_name else 'allwise_cntr'

        q = 'SELECT \n\t'
        for k in keys:
            q += f'{db_name}.{k}, '
        q += '\n\tmine.wise_id \n'
        q += f'FROM\n\t{db_name} \n'
        q += f'INNER JOIN\n\tTAP_UPLOAD.ids AS mine ON {db_name}.{id_key} = mine.WISE_id \n'
        q += 'WHERE \n'
        for c in self.constraints:
            q += f'\t{db_name}.{c} and \n'
        q = q.strip(" and \n")
        logger.debug(f"\n{q}")
        return q

    def _query_for_photometry(self, tables):

        # only integers can be uploaded
        wise_id = [int(idd) for idd in self.parent_sample.df[self.parent_wise_source_id_key]]
        upload_table = Table({'wise_id': wise_id})

        # ----------------------------------------------------------------------
        #     Do the query
        # ----------------------------------------------------------------------

        jobs = dict()
        for t in np.atleast_1d(tables):
            qstring = self._get_photometry_query_string(t)
            job = WISEData.service.submit_job(qstring, uploads={'ids': upload_table})
            job.run()
            logger.info(f'submitted job for {t}: ')
            logger.debug(f'Job: {job.url}; {job.phase}')
            jobs[t] = job

        lightcurves = list()
        for t, job in jobs.items():
            logger.info(f"Waiting on query of {t}")
            logger.info(" ........")
            job.wait()

            logger.info('Done!')
            lightcurve = job.fetch_result().to_table().to_pandas()
            lightcurves.append(lightcurve.rename(columns=self.photometry_table_keymap[t]))

        self._all_lightcurves = pd.concat(lightcurves)

    def _select_individual_lightcurves_and_bin(self, drop=True):

        # ----------------------------------------------------------------------
        #     select individual lightcurves
        # ----------------------------------------------------------------------

        logger.info('selecting individual lightcurves and bin ...')
        self.binned_lightcurves = dict()
        for ID in self._all_lightcurves.wise_id.unique():
            m = self._all_lightcurves.wise_id == ID
            lightcurve = self._all_lightcurves[m]
            if drop:
                # drop the selected rows from self._all_lightcurves to save memory
                self._all_lightcurves.drop(self._all_lightcurves.index[m], inplace=True)

        # ----------------------------------------------------------------------
        #     bin lightcurves
        # ----------------------------------------------------------------------

            # bin lightcurves in time intervals where observations are closer than 100 days together
            sorted_mjds = np.sort(lightcurve.mjd)
            epoch_bounds_mask = (sorted_mjds[1:] - sorted_mjds[:-1]) > 100
            epoch_bounds = np.array(
                [lightcurve.mjd.min()] +
                list(sorted_mjds[1:][epoch_bounds_mask]) +
                [lightcurve.mjd.max()*1.01]  # this just makes sure that the last datapoint gets selected as well
            )
            epoch_intervals = np.array([epoch_bounds[:-1], epoch_bounds[1:]]).T

            binned_lc = pd.DataFrame()
            for ei in epoch_intervals:
                r = dict()
                epoch_mask = (lightcurve.mjd >= ei[0]) & (lightcurve.mjd < ei[1])
                r['mean_mjd'] = np.median(lightcurve.mjd[epoch_mask])

                for b in self.bands:
                    f = lightcurve[f"{b}{self.flux_key_ext}"][epoch_mask]
                    mean = np.average(f, weights=lightcurve[f"{b}{self.error_key_ext}"][epoch_mask])
                    u = np.sqrt(sum((f - mean) ** 2) / len(f))
                    r[f'{b}_mean_flux'] = mean
                    r[f'{b}_flux_rms'] = u

                binned_lc = binned_lc.append(r, ignore_index=True)

            self.binned_lightcurves[int(ID)] = binned_lc.to_dict()

    def _cache_binned_lightcurves(self):
        # ----------------------------------------------------------------------
        #     save to JSON
        # ----------------------------------------------------------------------
        logger.debug(f"saving to {self._cached_binned_lcs_json}")
        with open(self._cached_binned_lcs_json, "w") as f:
            json.dump(self.binned_lightcurves, f)

    #################################
    # END GET PHOTOMETRY DATA       #
    ###########################################################################################################


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--logging_level', type=str, default='INFO')
    cfg = parser.parse_args()

    main_logger.setLevel(cfg.logging_level)

    wise_data = WISEData()
    wise_data.match_all_chunks()
    wise_data.get_photometric_data()