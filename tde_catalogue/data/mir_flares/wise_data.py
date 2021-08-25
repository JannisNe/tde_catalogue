import os, subprocess, copy, json, argparse, tqdm, time
import multiprocessing as mp
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
        self.n_chunks = n_chunks

        # set up parent sample keys
        self.parent_ra_key = parent_sample.default_keymap['ra']
        self.parent_dec_key = parent_sample.default_keymap['dec']
        self.parent_wise_source_id_key = 'WISE_id'
        self.parent_sample_wise_skysep_key = 'sep_to_WISE_source'

        # set up directories
        self.cache_dir = os.path.join(cache_dir, base_name)
        self._cache_photometry_dir = os.path.join(self.cache_dir, "photometry")
        self.output_dir = os.path.join(output_dir, base_name)
        self.lightcurve_dir = os.path.join(self.output_dir, "lightcurves")
        self.plots_dir = os.path.join(plots_dir, base_name)

        for d in [self.cache_dir, self._cache_photometry_dir, self.output_dir, self.lightcurve_dir, self.plots_dir]:
            if not os.path.isdir(d):
                os.makedirs(d)

        # set up result attributes
        self._split_photometry_key = '__chunk'
        self._cached_raw_photometry_prefix = 'raw_photometry'
        self._all_lightcurves = None
        self._cached_all_lightcurves = None
        self.binned_lightcurves = None
        self._cached_binned_lcs_json = os.path.join(self.cache_dir, "binned_lcs.json")

        #########################
        # END SET-UP            #
        #######################################################################################

        #######################################################################################
        # START CHUNK MASK      #
        #########################

        self.parent_sample = parent_sample
        min_dec = min(self.parent_sample.df[self.parent_dec_key])
        max_dec = max(self.parent_sample.df[self.parent_dec_key])
        logger.info(f'Declination: ({min_dec}, {max_dec})')
        self.parent_sample.df[self.parent_wise_source_id_key] = ""
        self.parent_sample.df[self.parent_sample_wise_skysep_key] = np.inf

        sin_bounds = np.linspace(np.sin(np.radians(min_dec * 1.001)), np.sin(np.radians(max_dec)), n_chunks+1, endpoint=True)
        self.dec_intervalls = np.degrees(np.arcsin(np.array([sin_bounds[:-1], sin_bounds[1:]]).T))
        logger.info(f'Declination intervalls are {self.dec_intervalls}')

        self.dec_interval_masks = list()
        for i, dec_intervall in enumerate(self.dec_intervalls):
            dec_intervall_mask = (self.parent_sample.df[self.parent_dec_key] > min(dec_intervall)) & \
                                 (self.parent_sample.df[self.parent_dec_key] <= max(dec_intervall))
            if not np.any(dec_intervall_mask):
                logger.warning(f"No objects selected in chunk {i}!")
            self.dec_interval_masks.append(dec_intervall_mask)

        _is_not_selected = sum(self.dec_interval_masks) != 1
        if np.any(_is_not_selected):
            raise ValueError(f"{len(self.dec_interval_masks[0][_is_not_selected])} objects not selected!"
                             f"Index: {np.where(_is_not_selected)[0]}")

        #########################
        # END CHUNK MASK        #
        #######################################################################################

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
        self.parent_sample.save_local()

    def match_single_chunk(self, chunk_number,
                           table_name="AllWISE Source Catalog"):
        """
        Match the parent sample to WISE
        :param chunk_number: int, number of the declination chunk
        :param table_name: str, optional, WISE table to match to, default is AllWISE Source Catalog
        """

        dec_intervall_mask = self.dec_interval_masks[chunk_number]
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
        _output_file = os.path.join(self.cache_dir, f"parent_sample_chunk{chunk_number}.tbl")
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

    def get_photometric_data(self, tables=None, perc=1):
        """
        Load photometric data from the IRSA server for the matched sample
        :param tables: list like, WISE tables to use for photometry query, defaults to AllWISE and NOEWISER photometry
        :param perc: float, percentage of sources to load photometry for, default 1
        """

        if tables is None:
            tables = [
                'NEOWISE-R Single Exposure (L1b) Source Table',
                'AllWISE Multiepoch Photometry Table'
            ]

        self._query_for_photometry(tables, perc)
        self._select_individual_lightcurves_and_bin()

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

    def _chunk_photometry_cache_filename(self, table_nice_name, chunk_number):
        table_name = self.get_db_name(table_nice_name)
        fn = f"{self._cached_raw_photometry_prefix}{table_name}{self._split_photometry_key}{chunk_number}.csv"
        return os.path.join(self._cache_photometry_dir, fn)

    def _query_for_photometry(self, tables, perc):

        # only integers can be uploaded
        wise_id = np.array([int(idd) for idd in self.parent_sample.df[self.parent_wise_source_id_key]])
        logger.debug(f"{len(wise_id)} IDs in total")

        # if perc is smaller than one select only a subset of wise IDs
        if perc < 1:
            logger.debug(f"Getting {perc:.2f} % of IDs")
            N_ids = int(round(len(wise_id) * perc))
            wise_id = np.random.default_rng().choice(wise_id, N_ids, replace=False, shuffle=False)
            logger.debug(f"selected {len(wise_id)} IDs")
        upload_table = Table({'wise_id': wise_id})

    # ----------------------------------------------------------------------
    #     Do the query
    # ----------------------------------------------------------------------

        jobs = dict()
        for t in np.atleast_1d(tables):
            qstring = self._get_photometry_query_string(t)
            jobs[t] = dict()
            for i, m in enumerate(self.dec_interval_masks):
                job = WISEData.service.submit_job(qstring, uploads={'ids': upload_table[np.array(m)]})
                job.run()
                logger.info(f'submitted job for {t} for chunk {i}: ')
                logger.debug(f'Job: {job.url}; {job.phase}')
                jobs[t][i] = job

        _wait_for_h = 8
        logger.info(f"wait for {_wait_for_h} hours to give jobs some time")
        time.sleep(_wait_for_h * 3600)

        for t, jobd in jobs.items():
            for i, job in jobd.items():
                logger.info(f"Waiting on {i}th query of {t}")
                logger.info(" ........")
                # Sometimes a connection Error occurs.
                # In that case try again as long as job.wait() exits normally
                while True:
                    try:
                        job.wait()
                        break
                    except vo.dal.exceptions.DALServiceError as e:
                        logger.warning(f"DALServiceError: {e}")

                logger.info('Done!')
                lightcurve = job.fetch_result().to_table().to_pandas()
                fn = self._chunk_photometry_cache_filename(t, i)
                logger.debug(f"saving under {fn}")
                lightcurve.rename(columns=self.photometry_table_keymap[t]).to_csv(fn)

    # ----------------------------------------------------------------------
    #     select individual lightcurves and bin
    # ----------------------------------------------------------------------

    def _select_individual_lightcurves_and_bin(self, ncpu=15):
        logger.info('selecting individual lightcurves and bin ...')
        ncpu = min(self.n_chunks, ncpu)
        logger.debug(f"using {ncpu} CPUs")
        args = list(range(self.n_chunks))
        with mp.Pool(ncpu) as p:
            tqdm.tqdm(p.imap(self._subprocess_select_and_bin, args), total=self.n_chunks, desc='select and bin')

    def _cache_chunk_binned_lightcurves_filename(self, chunk_number):
        fn = f"binned_lightcurves{self._split_photometry_key}{chunk_number}.json"
        return os.path.join(self._cache_photometry_dir, fn)

    def _save_chunk_binned_lcs(self, chunk_number, binned_lcs):
        fn = self._cache_chunk_binned_lightcurves_filename(chunk_number)
        with open(fn, "w") as f:
            json.dump(binned_lcs, f)

    def _load_chunk_binned_lcs(self, chunk_number):
        fn = self._cache_chunk_binned_lightcurves_filename(chunk_number)
        with open(fn, "r") as f:
            binned_lcs = json.load(f)
        return binned_lcs

    def _subprocess_select_and_bin(self, chunk_number):

        # load only the files for this chunk
        fns = [fn for fn in os.listdir(self._cache_photometry_dir)
               if (fn.startswith(self._cached_raw_photometry_prefix) and fn.endswith(str(chunk_number)))]
        lightcurves = pd.concat([pd.read_csv(fn) for fn in fns])

        # run through the ids and bin the lightcurves
        unique_id = lightcurves.wise_id.unique()
        binned_lcs = dict()
        for ID in unique_id:
            m = self._all_lightcurves.wise_id == ID
            lightcurve = self._all_lightcurves[m]

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

            binned_lcs[ID] = binned_lc.to_dict()

        self._save_chunk_binned_lcs(chunk_number, binned_lcs)

    #################################
    # END GET PHOTOMETRY DATA       #
    ###########################################################################################################


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--logging_level', type=str, default='INFO')
    parser.add_argument('--phot', type=bool, default=False, nargs='?', const=True)
    cfg = parser.parse_args()

    main_logger.setLevel(cfg.logging_level)

    wise_data = WISEData()
    wise_data.match_all_chunks()
    if cfg.phot:
        wise_data.get_photometric_data()