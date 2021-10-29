import os, subprocess, copy, json, argparse, tqdm, time, threading, math, queue, pickle, getpass, requests
import multiprocessing as mp
import pandas as pd
import numpy as np
import pyvo as vo
import astropy.units as u
from astropy.table import Table
import matplotlib.pyplot as plt

from tde_catalogue import main_logger, cache_dir, plots_dir, output_dir, BASHFILE
from tde_catalogue.data.mir_flares import base_name as mir_base_name
from tde_catalogue.data.mir_flares.combined_parent_sample import CombinedParentSample
from tde_catalogue.data.mir_flares.sdss_photometric_galaxies import SDSSPhotometricGalaxies

logger = main_logger.getChild(__name__)


class WISEData:

    base_name = mir_base_name + '/WISE_data'
    service_url = 'https://irsa.ipac.caltech.edu/TAP'
    service = vo.dal.TAPService(service_url)
    active_tap_phases = {"QUEUED", "EXECUTING", "RUN", "COMPLETED", "ERROR", "UNKNOWN"}
    running_tap_phases = ["QUEUED", "EXECUTING", "RUN"]
    done_tap_phases = {"COMPLETED", "ABORTED", "ERROR"}
    status_cmd = f'qstat -u {getpass.getuser()}'


    table_names = pd.DataFrame([
        ('AllWISE Multiepoch Photometry Table', 'allwise_p3as_mep'),
        ('AllWISE Source Catalog', 'allwise_p3as_psd'),
        ('WISE 3-Band Cryo Single Exposure (L1b) Source Table', 'allsky_3band_p1bs_psd'),
        ('NEOWISE-R Single Exposure (L1b) Source Table', 'neowiser_p1bs_psd'),

    ], columns=['nice_table_name', 'table_name'])

    bands = ['W1', 'W2']
    flux_key_ext = "_flux"
    mag_key_ext = "_mag"
    error_key_ext = "_error"
    band_plot_colors = {'W1': 'r', 'W2': 'b'}

    mean_key = '_mean'
    rms_key = '_rms'
    upper_limit_key = '_ul'

    photometry_table_keymap = {
        'AllWISE Multiepoch Photometry Table': {
            'flux': {
                'w1flux_ep':    f'W1{flux_key_ext}',
                'w1sigflux_ep': f'W1{flux_key_ext}{error_key_ext}',
                'w2flux_ep':    f'W2{flux_key_ext}',
                'w2sigflux_ep': f'W2{flux_key_ext}{error_key_ext}'
            },
            'mag': {
                'w1mpro_ep':    f'W1{mag_key_ext}',
                'w1sigmpro_ep': f'W1{mag_key_ext}{error_key_ext}',
                'w2mpro_ep':    f'W2{mag_key_ext}',
                'w2sigmpro_ep': f'W2{mag_key_ext}{error_key_ext}'
            }
        },
        'NEOWISE-R Single Exposure (L1b) Source Table': {
            'flux': {
                'w1flux':       f'W1{flux_key_ext}',
                'w1sigflux':    f'W1{flux_key_ext}{error_key_ext}',
                'w2flux':       f'W2{flux_key_ext}',
                'w2sigflux':    f'W2{flux_key_ext}{error_key_ext}'
            },
            'mag': {
                'w1mpro':       f'W1{mag_key_ext}',
                'w1sigmpro':    f'W1{mag_key_ext}{error_key_ext}',
                'w2mpro':       f'W2{mag_key_ext}',
                'w2sigmpro':    f'W2{mag_key_ext}{error_key_ext}'
            }
        }
    }

    constraints = [
        "nb < 2",
        "na < 1",
        "cc_flags like '00%'",
        "qi_fact >= 1",
        "saa_sep >= 5",
        "moon_masked like '00%'"
    ]

    def __init__(self,
                 min_sep_arcsec=10,
                 n_chunks=2,
                 base_name=base_name,
                 parent_sample_class=SDSSPhotometricGalaxies):
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

        self.parent_sample_class = parent_sample_class
        parent_sample = parent_sample_class() if parent_sample_class else None
        self.base_name = base_name
        self.min_sep = min_sep_arcsec * u.arcsec
        self._n_chunks = n_chunks

        # --------------------------- vvvv set up parent sample vvvv --------------------------- #
        self.parent_ra_key = parent_sample.default_keymap['ra'] if parent_sample else None
        self.parent_dec_key = parent_sample.default_keymap['dec'] if parent_sample else None
        self.parent_wise_source_id_key = 'AllWISE_id'
        self.parent_sample_wise_skysep_key = 'sep_to_WISE_source'
        self.parent_sample_default_entries = {
            self.parent_wise_source_id_key: "",
            self.parent_sample_wise_skysep_key: np.inf
        }

        self.parent_sample = parent_sample

        if self.parent_sample:
            for k, default in self.parent_sample_default_entries.items():
                if k not in parent_sample.df.columns:
                    self.parent_sample.df[k] = default

            self._no_allwise_source = self.parent_sample.df[self.parent_sample_wise_skysep_key] == np.inf

        else:
            self._no_allwise_source = None
        # --------------------------- ^^^^ set up parent sample ^^^^ --------------------------- #

        # set up directories
        self.cache_dir = os.path.join(cache_dir, base_name)
        self._cache_photometry_dir = os.path.join(self.cache_dir, "photometry")
        self.cluster_dir = os.path.join(self.cache_dir, 'cluster')
        self.cluster_log_dir = os.path.join(self.cluster_dir, 'logs')
        self.output_dir = os.path.join(output_dir, base_name)
        self.lightcurve_dir = os.path.join(self.output_dir, "lightcurves")
        self.plots_dir = os.path.join(plots_dir, base_name)

        for d in [self.cache_dir, self._cache_photometry_dir, self.cluster_dir, self.cluster_log_dir,
                  self.output_dir, self.lightcurve_dir, self.plots_dir]:
            if not os.path.isdir(d):
                os.makedirs(d)

        self.submit_file = os.path.join(self.cluster_dir, 'submit.txt')

        # set up result attributes
        self._split_chunk_key = '__chunk'
        self._cached_raw_photometry_prefix = 'raw_photometry'
        self.tap_jobs = None
        self.queue = queue.Queue()
        self.clear_unbinned_photometry_when_binning = False

        self._tap_wise_id_key = 'wise_id'
        self._tap_orig_id_key = 'orig_id'

        #########################
        # END SET-UP            #
        #######################################################################################

        #######################################################################################
        # START CHUNK MASK      #
        #########################

        self.chunk_map = None
        self.n_chunks = self._n_chunks

        # set up cluster stuff
        self.job_id = None
        self._n_cluster_jobs_per_chunk = None
        self.cluster_jobID_map = None
        self.clusterJob_chunk_map = None
        self.cluster_info_file = os.path.join(self.cluster_dir, 'cluster_info.pkl')

    @property
    def n_chunks(self):
        return self._n_chunks

    @n_chunks.setter
    def n_chunks(self, value):
        """Sets the private variable _n_chunks and re-calculates the declination interval masks"""

        if value > 50:
            logger.warning(f"Very large number of chunks ({value})! "
                           f"Pay attention when getting photometry to not kill IRSA!")

        self._n_chunks = value

        if self.parent_sample:

            self.chunk_map = np.zeros(len(self.parent_sample.df))
            N_in_chunk = int(round(len(self.chunk_map) / self._n_chunks))
            for i in range(self._n_chunks):
                start_ind = i * N_in_chunk
                end_ind = start_ind + N_in_chunk
                self.chunk_map[start_ind:end_ind] = int(i)

        else:
            logger.warning("No parent sample given! Can not calculate dec interval masks!")

    def _get_chunk_number(self, wise_id=None, parent_sample_index=None):
        if (not wise_id) and (not parent_sample_index):
            raise Exception

        if wise_id:
            parent_sample_index = np.where(self.parent_sample.df[self.parent_wise_source_id_key] == int(wise_id))[0]
            logger.debug(f"wise ID {wise_id} at index {parent_sample_index}")

        _chunk_number = int(self.chunk_map[int(parent_sample_index)])
        logger.debug(f"chunk number is {_chunk_number} for {parent_sample_index}")
        return _chunk_number

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

    def match_all_chunks(self, table_name="AllWISE Source Catalog", rematch=False, rematch_to_neowise=False, save_when_done=True):
        for i in range(self.n_chunks):
            self._match_single_chunk(i, table_name)

        _dupe_mask = self._get_dubplicated_wise_id_mask()
        if np.any(_dupe_mask) and rematch:
            self._rematch_duplicates(table_name, _dupe_mask, filext="_rematch1")

            _inf_mask = ~(self.parent_sample.df[self.parent_sample_wise_skysep_key] < np.inf)
            if np.any(_inf_mask) and rematch_to_neowise:
                logger.info(f"Still {len(self.parent_sample.df[_inf_mask])} entries without match."
                            f"Looking in NoeWISE Photometry")
                for mi in range(self.n_chunks):
                    m = self.chunk_map == mi
                    _interval_inf_mask = _inf_mask & m
                    if np.any(_interval_inf_mask):
                        self._rematch_duplicates(table_name='NEOWISE-R Single Exposure (L1b) Source Table',
                                                 mask=_interval_inf_mask,
                                                 filext=f"_rematch2_c{mi}")

        self._no_allwise_source = self.parent_sample.df[self.parent_sample_wise_skysep_key] == np.inf
        if np.any(self._no_allwise_source):
            logger.warning(f"{len(self.parent_sample.df[self._no_allwise_source])} of {len(self.parent_sample.df)} "
                           f"entries without match!")

        if np.any(self._get_dubplicated_wise_id_mask()):
            logger.warning(self.parent_sample.df[self._get_dubplicated_wise_id_mask()])

        if save_when_done:
            self.parent_sample.save_local()

    def _run_gator_match(self, in_file, out_file, table_name,
                         one_to_one=True, minsep_arcsec=None, additional_keys='', silent=False, constraints=None):
        _one_to_one = '-F one_to_one=1 ' if one_to_one else ''
        _minsep_arcsec = self.min_sep.to("arcsec").value if minsep_arcsec is None else minsep_arcsec
        _db_name = self.get_db_name(table_name)
        _silent = "-s " if silent else ""
        _constraints = '-F constraints="' + " and ".join(constraints).replace('%', '%%') + '" ' if constraints else ""

        if _db_name == "allwise_p3as_mep":
            _sigpos = _source_id = _des = ""
            _id_key = "cntr_mf,cntr"
        else:
            _sigpos = 'sigra,sigdec,'
            _source_id = "source_id,"
            _des = 'designation,' if 'allwise' in _db_name else ''
            _id_key = 'cntr' if 'allwise' in _db_name else 'allwise_cntr,cntr'

        submit_cmd = f'curl ' \
                     f'--connect-timeout 3600 ' \
                     f'--max-time 3600 ' \
                     f'{_silent}' \
                     f'-o {out_file} ' \
                     f'-F filename=@{in_file} ' \
                     f'-F catalog={_db_name} ' \
                     f'-F spatial=Upload ' \
                     f'-F uradius={_minsep_arcsec} ' \
                     f'-F outfmt=1 ' \
                     f'{_one_to_one}' \
                     f'{_constraints}' \
                     f'-F selcols={_des}{_source_id}ra,dec,{_sigpos}{_id_key}{additional_keys} ' \
                     f'"https://irsa.ipac.caltech.edu/cgi-bin/Gator/nph-query"'

        logger.debug(f'submit command: {submit_cmd}')
        N_tries = 10
        while True:
            try:
                process = subprocess.Popen(submit_cmd, stdout=subprocess.PIPE, shell=True)
                break
            except OSError as e:
                if N_tries < 1:
                    raise OSError(e)
                logger.warning(f"{e}, retry")
                N_tries -= 1

        out_msg, err_msg = process.communicate()
        if out_msg:
            logger.info(out_msg.decode())
        if err_msg:
            logger.error(err_msg.decode())

        if os.path.isfile(out_file):
            return 1
        else:
            return 0

    def _match_to_wise(self, in_filename, out_filename, mask, table_name,
                       # remove_when_done=True,
                       N_retries=10, **gator_kwargs):
        selected_parent_sample = copy.copy(
            self.parent_sample.df.loc[mask, [self.parent_ra_key, self.parent_dec_key]])
        selected_parent_sample.rename(columns={self.parent_dec_key: 'dec',
                                               self.parent_ra_key: 'ra'},
                                      inplace=True)
        logger.debug(f"{len(selected_parent_sample)} selected")

        # write to IPAC formatted table
        _selected_parent_sample_astrotab = Table.from_pandas(selected_parent_sample, index=True)
        logger.debug(f"writing {len(_selected_parent_sample_astrotab)} "
                     f"objects to {in_filename}")
        _selected_parent_sample_astrotab.write(in_filename, format='ipac', overwrite=True)
        _done = False

        while True:
            if N_retries == 0:
                raise RuntimeError('Failed with retries')

            try:
                # use Gator to query IRSA
                success = self._run_gator_match(in_filename, out_filename, table_name, **gator_kwargs)

                if not success:
                    # if not successful try again
                    logger.warning("no success, try again")
                    continue

                # load the result file
                gator_res = Table.read(out_filename, format='ipac')
                logger.debug(f"found {len(gator_res)} results")
                return gator_res

            except ValueError:
                # this will happen if the gator match returns an output containing the error message
                # read and display error message, then try again
                with open(out_filename, 'r') as f:
                    err_msg = f.read()
                logger.warning(f"{err_msg}: try again")

            finally:
                N_retries -= 1

    def _match_single_chunk(self, chunk_number, table_name):
        """
        Match the parent sample to WISE
        :param chunk_number: int, number of the declination chunk
        :param table_name: str, optional, WISE table to match to, default is AllWISE Source Catalog
        """

        dec_intervall_mask = self.chunk_map == chunk_number
        logger.debug(f"Any selected: {np.any(dec_intervall_mask)}")
        _parent_sample_declination_band_file = os.path.join(self.cache_dir, f"parent_sample_chunk{chunk_number}.xml")
        _output_file = os.path.join(self.cache_dir, f"parent_sample_chunk{chunk_number}.tbl")
        gator_res = self._match_to_wise(
            in_filename=_parent_sample_declination_band_file,
            out_filename=_output_file,
            mask=dec_intervall_mask,
            table_name=table_name
        )

        for fn in [_parent_sample_declination_band_file, _output_file]:
            try:
                logger.debug(f"removing {fn}")
                os.remove(fn)
            except FileNotFoundError:
                logger.warning(f"No File!!")

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

        _no_match_mask = self.parent_sample.df[self.parent_sample_wise_skysep_key].isna() & dec_intervall_mask
        for k, default in self.parent_sample_default_entries.items():
            self.parent_sample.df.loc[_no_match_mask, k] = default

    def _get_dubplicated_wise_id_mask(self):
        idf_sorted_sep = self.parent_sample.df.sort_values(self.parent_sample_wise_skysep_key)
        idf_sorted_sep['duplicate'] = idf_sorted_sep[self.parent_wise_source_id_key].duplicated(keep='first')
        idf_sorted_sep.sort_index(inplace=True)
        _inf_mask = idf_sorted_sep[self.parent_sample_wise_skysep_key] < np.inf
        _dupe_mask = idf_sorted_sep['duplicate'] & (_inf_mask)
        if np.any(_dupe_mask):
            _N_dupe = len(self.parent_sample.df[_dupe_mask])
            logger.info(f"{_N_dupe} duplicated entries in parent sample")
        return _dupe_mask

    def _rematch_duplicates(self, table_name, mask=None, filext=""):
        if mask is None:
            mask = self._get_dubplicated_wise_id_mask()

        for k, default in self.parent_sample_default_entries.items():
            self.parent_sample.df.loc[mask, k] = default

        _dupe_infile = os.path.join(self.cache_dir, f"parent_sample_duplicated{filext}.xml")
        _dupe_output_file = os.path.join(self.cache_dir, f"parent_sample_duplicated{filext}.tbl")
        _gator_res = self._match_to_wise(
            in_filename=_dupe_infile,
            out_filename=_dupe_output_file,
            mask=mask,
            table_name=table_name,
            one_to_one=False
        ).to_pandas()

        if len(_gator_res) == 0:
            logger.debug('No additional matches found')
            return

        _dupe_inds = list(self.parent_sample.df[mask].index)
        for in_id in tqdm.tqdm(_dupe_inds, desc='finding match for duplicates'):
            _m = _gator_res.index_01 == in_id
            _sorted_sel = _gator_res[_m].sort_values('dist_x')
            for _, r in _sorted_sel.iterrows():
                _wise_id = r['cntr' if 'allwise' in self.get_db_name(table_name) else 'allwise_cntr']
                _skysep = r['dist_x']
                __m = self.parent_sample.df[self.parent_wise_source_id_key] == _wise_id

                if not np.any(__m):
                    self.parent_sample.df.loc[in_id, self.parent_sample_wise_skysep_key] = _skysep
                    self.parent_sample.df.loc[in_id, self.parent_wise_source_id_key] = _wise_id
                    break

                else:
                    __msep = self.parent_sample.df[self.parent_sample_wise_skysep_key][__m] > _skysep
                    if np.any(__msep):
                        columns = ['ra', 'dec', self.parent_sample_wise_skysep_key, self.parent_wise_source_id_key]
                        txt = f"WISE ID: {_wise_id} with skysep {_skysep}:\n" \
                              f"{self.parent_sample.df.loc[in_id].to_string()}" \
                              f" \n" \
                              f"{self.parent_sample.df.loc[__m].to_string(columns=columns)}"
                        if len(self.parent_sample.df[self.parent_wise_source_id_key][__m][__msep]) == 1:
                            logger.warning(txt)
                            self.parent_sample.df.loc[in_id, self.parent_sample_wise_skysep_key] = _skysep
                            self.parent_sample.df.loc[in_id, self.parent_wise_source_id_key] = _wise_id
                            for k in [self.parent_wise_source_id_key, self.parent_sample_wise_skysep_key]:
                                self.parent_sample.df.loc[np.where(__m)[0][__msep], k] = self.parent_sample_default_entries[k]
                        else:
                            raise Exception(txt)

    ###################################################
    # END MATCH PARENT SAMPLE TO WISE SOURCES         #
    ###########################################################################################################

    ###########################################################################################################
    # START GET PHOTOMETRY DATA       #
    ###################################

    def get_photometric_data(self, tables=None, perc=1, wait=5, service=None, mag=True, flux=False, nthreads=100,
                             chunks=None, cluster_jobs_per_chunk=0):
        """
        Load photometric data from the IRSA server for the matched sample
        :param cluster_jobs_per_chunk: int, if greater than zero uses the DESY cluster
        :param tables: list like, WISE tables to use for photometry query, defaults to AllWISE and NOEWISER photometry
        :param perc: float, percentage of sources to load photometry for, default 1
        :param use_cluster: bool, submits to DESY cluster if True
        :param nthreads: int, max number of threads to launch
        :param flux: bool, get flux values if True
        :param mag: bool, gets magnitude values if True
        :param service: str, either of 'gator' or 'tap', selects base on elements per chunk by default
        :param wait: float, time in hours to wait after submitting TAP jobs
        :param chunks: list-like, containing indices of chunks to download
        """

        if tables is None:
            tables = [
                'AllWISE Multiepoch Photometry Table',
                'NEOWISE-R Single Exposure (L1b) Source Table'
            ]

        if chunks is None:
            chunks = list(range(round(int(self.n_chunks * perc))))

        if service is None:
            elements_per_chunk = len(self.parent_sample.df) / self.n_chunks
            service = 'tap' if elements_per_chunk > 300 else 'gator'

        logger.debug(f"Getting {perc * 100:.2f}% of lightcurve chunks ({len(chunks)}) via {service} "
                     f"in {'magnitude' if mag else ''} {'flux' if flux else ''} "
                     f"from {tables}")

        if cluster_jobs_per_chunk:
            self.n_cluster_jobs_per_chunk = cluster_jobs_per_chunk

            cluster_time_s = len(self.parent_sample.df) / self._n_chunks / self.n_cluster_jobs_per_chunk
            if cluster_time_s > 24 * 3600:
                raise ValueError(f"cluster time per job would be longer than 24h! "
                                 f"Choose more than {self.n_cluster_jobs_per_chunk} jobs per chunk!")

            cluster_time = time.strftime('%H:%M:%S', time.gmtime(cluster_time_s))
            cluster_args = [1, cluster_time, '10', service]

        if service == 'tap':
            self._query_for_photometry(tables, chunks, wait, mag, flux, nthreads)
            if cluster_jobs_per_chunk:
                pass
                # self.run_cluster(*cluster_args)

        elif service == 'gator':
            if cluster_jobs_per_chunk:
                self.run_cluster(*cluster_args)
            else:
                self._query_for_photometry_gator(tables, chunks, mag, flux, nthreads)

        if not cluster_jobs_per_chunk:
            self._select_individual_lightcurves_and_bin(service=service, chunks=chunks)
        self._combine_binned_lcs(service)

    def _lightcurve_filename(self, service, chunk_number=None, jobID=None):
        if (chunk_number is None) and (jobID is None):
            return os.path.join(self.lightcurve_dir, f"binned_lightcurves_{service}.json")
        else:
            fn = f"binned_lightcurves_{service}{self._split_chunk_key}{chunk_number}"
            if (chunk_number is not None) and (jobID is None):
                return os.path.join(self._cache_photometry_dir, fn + ".json")
            else:
                return os.path.join(self._cache_photometry_dir, fn + f"_{jobID}.json")

    def _load_lightcurves(self, service, chunk_number=None, jobID=None, remove=False):
        fn = self._lightcurve_filename(service, chunk_number, jobID)
        logger.debug(f"loading {fn}")
        try:
            with open(fn, "r") as f:
                lcs = json.load(f)
            if remove:
                logger.debug(f"removing {fn}")
                os.remove(fn)
            return lcs
        except FileNotFoundError:
            logger.warning(f"No file {fn}")

    def _save_lightcurves(self, lcs, service, chunk_number=None, jobID=None, overwrite=False):
        fn = self._lightcurve_filename(service, chunk_number, jobID)
        logger.debug(f"saving {len(lcs)} new lightcurves to {fn}")

        if not overwrite:
            try:
                old_lcs = self._load_lightcurves(service=service, chunk_number=chunk_number, jobID=jobID)
                logger.debug(f"Found {len(old_lcs)}. Combining")
                lcs = lcs.update(old_lcs)
            except FileNotFoundError as e:
                logger.info(f"FileNotFoundError: {e}. Making new binned lightcurves.")

        with open(fn, "w") as f:
            json.dump(lcs, f)

    def load_binned_lcs(self, service):
        return self._load_lightcurves(service)

    def _combine_lcs(self, service=None, chunk_number=None, remove=False, overwrite=False):
        if not service:
            logger.info("Combining all lightcuves collected with all services")
            itr = ['service', ['gator', 'tap']]
            kwargs = {}
        elif chunk_number is None:
            logger.info(f"Combining all lightcurves collected with {service}")
            itr = ['chunk_number', range(self.n_chunks)]
            kwargs = {'service': service}
        elif chunk_number is not None:
            logger.info(f"Combining all lightcurves collected eith {service} for chunk {chunk_number}")
            itr = ['jobID',
                   list(self.clusterJob_chunk_map.index[self.clusterJob_chunk_map.chunk_number == chunk_number])]
            kwargs = {'service': service, 'chunk_number': chunk_number}
        else:
            raise NotImplementedError

        lcs = None
        for i in itr[1]:
            kw = dict(kwargs)
            kw[itr[0]] = i
            kw['remove'] = remove
            ilcs = self._load_lightcurves(**kw)
            if isinstance(lcs, type(None)):
                lcs = dict(ilcs)
            else:
                lcs.update(ilcs)

        self._save_lightcurves(lcs, service=service, chunk_number=chunk_number, overwrite=overwrite)

    # ----------------------------------------------------------------------------------- #
    # START using GATOR to get photometry        #
    # ------------------------------------------ #

    def _gator_chunk_photometry_cache_filename(self, table_nice_name, chunk_number,
                                               additional_neowise_query=False, gator_input=False):
        table_name = self.get_db_name(table_nice_name)
        _additional_neowise_query = '_neowise_gator' if additional_neowise_query else ''
        _gator_input = '_gator_input' if gator_input else ''
        _ending = '.xml' if gator_input else'.tbl'
        fn = f"{self._cached_raw_photometry_prefix}_{table_name}{_additional_neowise_query}{_gator_input}" \
             f"{self._split_chunk_key}{chunk_number}{_ending}"
        return os.path.join(self._cache_photometry_dir, fn)

    def _thread_query_photometry_gator(self, chunk_number, table_name, mag, flux):
        _infile = self._gator_chunk_photometry_cache_filename(table_name, chunk_number, gator_input=True)
        _outfile = self._gator_chunk_photometry_cache_filename(table_name, chunk_number)
        _nice_name = self.get_db_name(table_name, nice=True)
        _additional_keys_list = ['mjd']
        if mag:
            _additional_keys_list += list(self.photometry_table_keymap[_nice_name]['mag'].keys())
        if flux:
            _additional_keys_list += list(self.photometry_table_keymap[_nice_name]['flux'].keys())

        _additional_keys = "," + ",".join(_additional_keys_list)
        _deci_mask = self.chunk_map == chunk_number
        _mask = _deci_mask & (~self._no_allwise_source)

        res = self._match_to_wise(
            in_filename=_infile,
            out_filename=_outfile,
            mask=_mask,
            table_name=table_name,
            one_to_one=False,
            additional_keys=_additional_keys,
            minsep_arcsec=self.min_sep.to('arcsec').value,
            silent=True,
            constraints=self.constraints
        )

        os.remove(_infile)
        return res

    def _gator_photometry_worker_thread(self):
        while True:
            args = self.queue.get()
            logger.debug(f"{args}")
            self._thread_query_photometry_gator(*args)
            self.queue.task_done()
            logger.info(f"{self.queue.qsize()} tasks remaining")

    def _query_for_photometry_gator(self, tables, chunks, mag, flux, nthreads):
        nthreads = min(nthreads, len(chunks))
        logger.debug(f'starting {nthreads} workers')
        threads = [threading.Thread(target=self._gator_photometry_worker_thread, daemon=True) for _ in range(nthreads)]

        logger.debug(f"using {len(chunks)} chunks")
        self.queue = queue.Queue()
        for t in np.atleast_1d(tables):
            for i in chunks:
                self.queue.put([i, t, mag, flux])

        logger.info(f"added {self.queue.qsize()} tasks to queue")
        for t in threads:
            t.start()
        self.queue.join()
        self.queue = None

    def _get_unbinned_lightcurves_gator(self, chunk_number, clear=False):
        # load only the files for this chunk
        fns = [os.path.join(self._cache_photometry_dir, fn)
               for fn in os.listdir(self._cache_photometry_dir)
               if (fn.startswith(self._cached_raw_photometry_prefix) and
                   fn.endswith(f"{self._split_chunk_key}{chunk_number}.tbl"))
               ]

        logger.debug(f"chunk {chunk_number}: loading {len(fns)} files for chunk {chunk_number}")

        _data = list()
        for fn in fns:
            data_table = Table.read(fn, format='ipac').to_pandas()

            t = 'allwise_p3as_mep' if 'allwise' in fn else 'neowiser_p1bs_psd'
            nice_name = self.get_db_name(t, nice=True)
            cols = dict(self.photometry_table_keymap[nice_name]['mag'])
            cols.update(self.photometry_table_keymap[nice_name]['flux'])
            if 'allwise' in fn:
                cols['cntr_mf'] = 'allwise_cntr'

            data_table = data_table.rename(columns=cols)
            _data.append(data_table)

            if clear:
                os.remove(fn)

        lightcurves = pd.concat(_data)
        return lightcurves

    def _subprocess_select_and_bin_gator(self, chunk_number=None, jobID=None):
        binned_lcs = dict()
        lightcurves = self._get_unbinned_lightcurves_gator(chunk_number,
                                                           clear=self.clear_unbinned_photometry_when_binning)

        if jobID:
            _indices = np.where(self.cluster_jobID_map == jobID)[0]

        else:
            _indices = lightcurves['index_01'].unique()

        for parent_sample_idx in _indices:
            parent_sample_idx_mask = lightcurves['index_01'] == parent_sample_idx
            selected_data = lightcurves[parent_sample_idx_mask]

            lum_keys = [c for c in lightcurves.columns if ("W1" in c) or ("W2" in c)]
            lightcurve = selected_data[['mjd'] + lum_keys]
            binned_lc = self.bin_lightcurve(lightcurve)
            if self.parent_sample:
                wise_id = self.parent_sample.df.loc[int(parent_sample_idx), self.parent_wise_source_id_key]
            else:
                wise_id = None
            binned_lcs[f"{int(parent_sample_idx)}_{wise_id}"] = binned_lc.to_dict()

        logger.debug(f"chunk {chunk_number}: saving {len(binned_lcs.keys())} binned lcs")
        self._save_lightcurves(binned_lcs, service='gator', chunk_number=chunk_number, jobID=jobID, overwrite=True)

    # ------------------------------------------ #
    # END using GATOR to get photometry          #
    # ----------------------------------------------------------------------------------- #

    # ----------------------------------------------------------------------------------- #
    # START using TAP to get photometry        #
    # ---------------------------------------- #

    def _get_photometry_query_string(self, table_name, mag, flux):
        """
        Construct a query string to submit to IRSA
        :param table_name: str, table name
        :return: str
        """
        logger.debug(f"constructing query for {table_name}")
        db_name = self.get_db_name(table_name)
        nice_name = self.get_db_name(table_name, nice=True)
        lum_keys = list()
        if mag:
            lum_keys += list(self.photometry_table_keymap[nice_name]['mag'].keys())
        if flux:
            lum_keys += list(self.photometry_table_keymap[nice_name]['flux'].keys())
        keys = ['mjd'] + lum_keys
        id_key = 'cntr_mf' if 'allwise' in db_name else 'allwise_cntr'

        q = 'SELECT \n\t'
        for k in keys:
            q += f'{db_name}.{k}, '
        q += f'\n\tmine.{self._tap_wise_id_key}, mine.{self._tap_orig_id_key} \n'
        q += f'FROM\n\t{db_name} \n'
        q += f'INNER JOIN\n\tTAP_UPLOAD.ids AS mine ON {db_name}.{id_key} = mine.WISE_id \n'
        q += 'WHERE \n'
        for c in self.constraints:
            q += f'\t{db_name}.{c} and \n'
        q = q.strip(" and \n")
        logger.debug(f"\n{q}")
        return q

    def _submit_job_to_TAP(self, chunk_number, table_name, mag, flux):
        i = chunk_number
        t = table_name
        m = self.chunk_map == i

        # if perc is smaller than one select only a subset of wise IDs
        sel = self.parent_sample.df[np.array(m) & ~ self._no_allwise_source]
        wise_id_sel = np.array(sel[self.parent_wise_source_id_key]).astype(int)
        id_sel = np.array(sel.index).astype(int)
        del sel

        upload_table = Table({self._tap_wise_id_key: wise_id_sel, self._tap_orig_id_key: id_sel})
        logger.debug(f"{chunk_number}th query of {table_name}: uploading {len(upload_table)} objects.")
        qstring = self._get_photometry_query_string(t, mag, flux)

        try:
            job = WISEData.service.submit_job(qstring, uploads={'ids': upload_table})
            job.run()
            logger.info(f'submitted job for {t} for chunk {i}: ')
            logger.debug(f'Job: {job.url}; {job.phase}')
            self.tap_jobs[t][i] = job
            self.queue.put((t, i))
        except requests.exceptions.ConnectionError as e:
            logger.warning(f"{chunk_number}th query of {table_name}: Could not submit TAP job!\n"
                           f"{e}")

    def _chunk_photometry_cache_filename(self, table_nice_name, chunk_number, additional_neowise_query=False):
        table_name = self.get_db_name(table_nice_name)
        _additional_neowise_query = '_neowise_gator' if additional_neowise_query else ''
        fn = f"{self._cached_raw_photometry_prefix}_{table_name}{_additional_neowise_query}" \
             f"{self._split_chunk_key}{chunk_number}.csv"
        return os.path.join(self._cache_photometry_dir, fn)

    def _thread_wait_and_get_results(self, t, i):
        logger.info(f"Waiting on {i}th query of {t} ........")
        _job = self.tap_jobs[t][i]
        # Sometimes a connection Error occurs.
        # In that case try again until job.wait() exits normally
        while True:
            try:
                _job.wait()
                break
            except vo.dal.exceptions.DALServiceError as e:
                if '404 Client Error: Not Found for url' in str(e):
                    raise vo.dal.exceptions.DALServiceError(f'{i}th query of {t}: {e}')
                else:
                    logger.warning(f"{i}th query of {t}: DALServiceError: {e}")

        logger.info(f'{i}th query of {t}: Done!')
        lightcurve = _job.fetch_result().to_table().to_pandas()
        fn = self._chunk_photometry_cache_filename(t, i)
        logger.debug(f"{i}th query of {t}: saving under {fn}")
        cols = dict(self.photometry_table_keymap[t]['mag'])
        cols.update(self.photometry_table_keymap[t]['flux'])
        lightcurve.rename(columns=cols).to_csv(fn)
        return

    def _tap_photometry_worker_thread(self):
        while True:
            try:
                t, i = self.queue.get(block=False)
            except queue.Empty:
                logger.debug("No more tasks, exiting")
                break

            job = self.tap_jobs[t][i]

            while True:
                try:
                    job._update(timeout=600)
                    phase = job._job.phase
                    break
                except vo.dal.exceptions.DALServiceError as e:
                    if '404 Client Error: Not Found for url' in str(e):
                        raise vo.dal.exceptions.DALServiceError(f'{i}th query of {t}: {e}')
                    else:
                        logger.warning(f"{i}th query of {t}: DALServiceError: {e}; try again")

            if phase in self.running_tap_phases:
                self.queue.put((t, i))
                self.queue.task_done()

            elif phase in self.done_tap_phases:
                self._thread_wait_and_get_results(t, i)
                self.queue.task_done()
                logger.info(f'{self.queue.qsize()} tasks left')

            else:
                logger.warning(f'queue {i} of {t}: Job not active! Phase is {phase}')

            time.sleep(np.random.uniform(600))

        logger.debug("closing thread")

    def _run_tap_worker_threads(self, nthreads):
        threads = [threading.Thread(target=self._tap_photometry_worker_thread, daemon=True)
                   for _ in range(nthreads)]

        for t in threads:
            t.start()

        try:
            self.queue.join()
        except KeyboardInterrupt:
            pass

        logger.info('all tap_jobs done!')
        for i, t in enumerate(threads):
            logger.debug(f"{i}th thread alive: {t.is_alive()}")

        self.tap_jobs = None
        del threads

    def _query_for_photometry(self, tables, chunks, wait, mag, flux, nthreads):
        # ----------------------------------------------------------------------
        #     Do the query
        # ----------------------------------------------------------------------
        self.tap_jobs = dict()
        self.queue = queue.Queue()
        tables = np.atleast_1d(tables)

        for t in tables:
            self.tap_jobs[t] = dict()
            for i in chunks:
                self._submit_job_to_TAP(i, t, mag, flux)
                logger.info("waiting for 2 minutes")
                time.sleep(2*60)

        logger.info(f'added {self.queue.qsize()} tasks to queue')
        logger.info(f"wait for {wait} hours to give tap_jobs some time")
        time.sleep(wait * 3600)
        nthreads = min(len(tables) * len(chunks), nthreads)
        self._run_tap_worker_threads(nthreads)
        self.queue = None

    # ----------------------------------------------------------------------
    #     select individual lightcurves and bin
    # ----------------------------------------------------------------------

    def _select_individual_lightcurves_and_bin(self, ncpu=35, service='tap', chunks=None):
        logger.info('selecting individual lightcurves and bin ...')
        ncpu = min(self.n_chunks, ncpu)
        logger.debug(f"using {ncpu} CPUs")
        args = list(range(self.n_chunks)) if not chunks else chunks
        logger.debug(f"multiprocessing arguments: {args}")

        _fcts = {
            'gator': self._subprocess_select_and_bin_gator,
            'tap': self._subprocess_select_and_bin
        }
        fct = _fcts[service]

        while True:
            try:
                logger.debug(f'trying with {ncpu}')
                p = mp.Pool(ncpu)
                break
            except OSError as e:
                logger.warning(e)
                if ncpu == 1:
                    break
                ncpu = int(round(ncpu - 1))

        if ncpu > 1:
            r = list(tqdm.tqdm(
                p.imap(fct, args), total=self.n_chunks, desc='select and bin'
            ))
            p.close()
            p.join()
        else:
            r = list(map(fct, args))

    def _get_unbinned_lightcurves(self, chunk_number, clear=False):
        # load only the files for this chunk
        fns = [os.path.join(self._cache_photometry_dir, fn)
               for fn in os.listdir(self._cache_photometry_dir)
               if (fn.startswith(self._cached_raw_photometry_prefix) and fn.endswith(
                f"{self._split_chunk_key}{chunk_number}.csv"
            ))]
        logger.debug(f"chunk {chunk_number}: loading {len(fns)} files for chunk {chunk_number}")
        lightcurves = pd.concat([pd.read_csv(fn) for fn in fns])

        if clear:
            for fn in fns:
                os.remove(fn)

        return lightcurves

    def _subprocess_select_and_bin(self, chunk_number=None, jobID=None):
        # run through the ids and bin the lightcurves
        lightcurves = self._get_unbinned_lightcurves(chunk_number, clear=self.clear_unbinned_photometry_when_binning)

        if jobID:
            # _chunk_number = self.clusterJob_chunk_map.loc[jobID, 'chunk_number']
            indices = np.where(self.cluster_jobID_map == jobID)[0]
        else:
            indices = lightcurves['index_01'].unique()

        logger.debug(f"chunk {chunk_number}: going through {len(indices)} IDs")

        binned_lcs = dict()
        for parent_sample_entry_id in tqdm.tqdm(indices):
            m = lightcurves[self._tap_orig_id_key] == parent_sample_entry_id
            lightcurve = lightcurves[m]

            if len(lightcurve) < 1:
                logger.warning(f"No data for {parent_sample_entry_id}")
                continue

            ID = lightcurve[self._tap_wise_id_key].iloc[0]
            binned_lc = self.bin_lightcurve(lightcurve)
            binned_lcs[f"{int(parent_sample_entry_id)}_{int(ID)}"] = binned_lc.to_dict()

        logger.debug(f"chunk {chunk_number}: saving {len(binned_lcs.keys())} binned lcs")
        # self._save_chunk_binned_lcs(_chunk_number, 'tap', binned_lcs, jobID)
        self._save_lightcurves(binned_lcs, service='tap', chunk_number=chunk_number, jobID=jobID, overwrite=True)

    # ---------------------------------------- #
    # END using TAP to get photometry          #
    # ----------------------------------------------------------------------------------- #

    # ----------------------------------------------------------------------
    #     bin lightcurves
    # ----------------------------------------------------------------------
    def bin_lightcurve(self, lightcurve):
        # bin lightcurves in time intervals where observations are closer than 100 days together
        sorted_mjds = np.sort(lightcurve.mjd)
        epoch_bounds_mask = (sorted_mjds[1:] - sorted_mjds[:-1]) > 100
        epoch_bounds = np.array(
            [lightcurve.mjd.min()] +
            list(sorted_mjds[1:][epoch_bounds_mask]) +
            [lightcurve.mjd.max() * 1.01]  # this just makes sure that the last datapoint gets selected as well
        )
        epoch_intervals = np.array([epoch_bounds[:-1], epoch_bounds[1:]]).T

        binned_lc = pd.DataFrame()
        for ei in epoch_intervals:
            r = dict()
            epoch_mask = (lightcurve.mjd >= ei[0]) & (lightcurve.mjd < ei[1])
            r['mean_mjd'] = np.median(lightcurve.mjd[epoch_mask])

            for b in self.bands:
                for lum_ext in [self.flux_key_ext, self.mag_key_ext]:
                    try:
                        f = lightcurve[f"{b}{lum_ext}"][epoch_mask]
                        e = lightcurve[f"{b}{lum_ext}{self.error_key_ext}"][epoch_mask]
                        ulims = pd.isna(e)
                        ul = np.all(pd.isna(e))

                        if ul:
                            mean = np.mean(f)
                            u_mes = 0
                        else:
                            f = f[~ulims]
                            e = e[~ulims]
                            w = e / sum(e)
                            mean = np.average(f, weights=w)
                            u_mes = np.sqrt(sum(e ** 2 / len(e)))

                        u_rms = np.sqrt(sum((f - mean) ** 2) / len(f))
                        r[f'{b}{self.mean_key}{lum_ext}'] = mean
                        r[f'{b}{lum_ext}{self.rms_key}'] = max(u_rms, u_mes)
                        r[f'{b}{lum_ext}{self.upper_limit_key}'] = bool(ul)
                    except KeyError:
                        pass

            binned_lc = binned_lc.append(r, ignore_index=True)

        return binned_lc

    # ----------------------------------------------------------------------------------- #
    # START using cluster for downloading and binning      #
    # ---------------------------------------------------- #

    @staticmethod
    def _qstat_output(qstat_command):
        """return the output of the qstat_command"""
        # start a subprocess to query the cluster
        process = subprocess.Popen(qstat_command, stdout=subprocess.PIPE, shell=True)
        # read the output
        tmp = process.stdout.read().decode()
        return str(tmp)

    @staticmethod
    def get_ids(qstat_command):
        """Takes a command that queries the DESY cluster and returns a list of job IDs"""
        st = WISEData._qstat_output(qstat_command)
        # If the output is an empty string there are no tasks left
        if st == '':
            ids = list()
        else:
            # Extract the list of job IDs
            ids = np.array([int(s.split(' ')[2]) for s in st.split('\n')[2:-1]])
        return ids

    def _ntasks_from_qstat_command(self, qstat_command):
        """Returns the number of tasks from the output of qstat_command"""
        # get the output of qstat_command
        ids = self.get_ids(qstat_command)
        ntasks = 0 if len(ids) == 0 else len(ids[ids == self.job_id])
        return ntasks

    @property
    def ntasks_total(self):
        """Returns the total number of tasks"""
        return self._ntasks_from_qstat_command(self.status_cmd)

    @property
    def ntasks_running(self):
        """Returns the number of running tasks"""
        return self._ntasks_from_qstat_command(self.status_cmd + " -s r")

    def wait_for_job(self):
        if self.job_id:
            time.sleep(10)
            i = 31
            j = 6
            while self.ntasks_total != 0:
                if i > 3:
                    logger.info(f'{time.asctime(time.localtime())} - Job{self.job_id}:'
                                f' {self.ntasks_total} entries in queue. '
                                f'Of these, {self.ntasks_running} are running tasks, and '
                                f'{self.ntasks_total - self.ntasks_running} are tasks still waiting to be executed.')
                    i = 0
                    j += 1

                if j > 7:
                    logger.info(self._qstat_output(self.status_cmd))
                    j = 0

                time.sleep(30)
                i += 1

        else:
            logger.info(f'No Job ID!')

    @property
    def n_cluster_jobs_per_chunk(self):
        return self._n_cluster_jobs_per_chunk

    @n_cluster_jobs_per_chunk.setter
    def n_cluster_jobs_per_chunk(self, value):
        self._n_cluster_jobs_per_chunk = value

        if value:
            n_jobs = self.n_chunks * int(value)
            logger.debug(f'setting {n_jobs} jobs.')
            self.cluster_jobID_map = np.zeros(len(self.parent_sample.df), dtype=int)
            self.clusterJob_chunk_map = pd.DataFrame(columns=['chunk_number'])

            for chunk_number in range(self.n_chunks):
                indices = np.where(self.chunk_map == chunk_number)[0]
                N_inds_per_job = int(math.ceil(len(indices) / self._n_cluster_jobs_per_chunk))
                for j in range(self._n_cluster_jobs_per_chunk):
                    job_nr = chunk_number*self._n_cluster_jobs_per_chunk + j + 1
                    self.clusterJob_chunk_map.loc[job_nr] = [chunk_number]
                    start_ind = j * N_inds_per_job
                    end_ind = start_ind + N_inds_per_job
                    self.cluster_jobID_map[indices[start_ind:end_ind]] = job_nr

        else:
            logger.warning(f'Invalid value for n_cluster_jobs_per_chunk: {value}')

    def _get_chunk_number_for_job(self, jobID):
        chunk_number = self.clusterJob_chunk_map.loc[jobID, 'chunk_number']
        return chunk_number

    def _save_cluster_info(self):
        logger.debug(f"writing cluster info to {self.cluster_info_file}")
        with open(self.cluster_info_file, "wb") as f:
            pickle.dump((self.cluster_jobID_map, self.clusterJob_chunk_map), f)

    def _load_cluster_info(self):
        logger.debug(f"loading cluster info from {self.cluster_info_file}")
        with open(self.cluster_info_file, "rb") as f:
            self.cluster_jobID_map, self.clusterJob_chunk_map = pickle.load(f)

    def clear_cluster_log_dir(self):
        fns = os.listdir(self.cluster_log_dir)
        for fn in fns:
            os.remove(os.path.join(self.cluster_log_dir, fn))

    def _make_cluster_script(self, cluster_h, cluster_ram, tables, service):
        script_fn = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'bin_lightcurves.py')

        if tables:
            tables = np.atleast_1d(tables)
            tables = [self.get_db_name(t, nice=False) for t in tables]
            tables_str = f"--tables {' '.join(tables)} \n"
        else:
            tables_str = '\n'

        casjobs_pw = os.environ['CASJOBS_PW']

        text = "#!/bin/zsh \n" \
               "## \n" \
               "##(otherwise the default shell would be used) \n" \
               "#$ -S /bin/zsh \n" \
               "## \n" \
               "##(the running time for this job) \n" \
              f"#$ -l h_cpu={cluster_h} \n" \
               "#$ -l h_rss=" + str(cluster_ram) + "\n" \
               "## \n" \
               "## \n" \
               "##(send mail on job's abort) \n" \
               "#$ -m a \n" \
               "## \n" \
               "##(stderr and stdout are merged together to stdout) \n" \
               "#$ -j y \n" \
               "## \n" \
               "## name of the job \n" \
               "## -N TDE Catalogue download \n" \
               "## \n" \
               "##(redirect output to:) \n" \
               f"#$ -o /dev/null \n" \
               "## \n" \
               "sleep $(( ( RANDOM % 60 )  + 1 )) \n" \
               'exec > "$TMPDIR"/${JOB_ID}_${SGE_TASK_ID}_stdout.txt ' \
               '2>"$TMPDIR"/${JOB_ID}_${SGE_TASK_ID}_stderr.txt \n' \
              f'source {BASHFILE} \n' \
              f'export CASJOBS_PW={casjobs_pw} \n' \
               'tde_catalogue \n' \
               'export O=1 \n' \
              f'python {script_fn} ' \
               f'--logging_level DEBUG ' \
               f'--base_name {self.base_name} ' \
               f'--min_sep_arcsec {self.min_sep.to("arcsec").value} ' \
               f'--n_chunks {self._n_chunks} ' \
               f'--service {service} ' \
               f'--job_id $SGE_TASK_ID ' \
               f'{tables_str}' \
               'cp $TMPDIR/${JOB_ID}_${SGE_TASK_ID}_stdout.txt ' + self.cluster_log_dir + '\n' \
               'cp $TMPDIR/${JOB_ID}_${SGE_TASK_ID}_stderr.txt ' + self.cluster_log_dir + '\n '

        logger.debug(f"Submit file: \n {text}")
        logger.debug(f"Creating file at {self.submit_file}")

        with open(self.submit_file, "w") as f:
            f.write(text)

        cmd = "chmod +x " + self.submit_file
        os.system(cmd)

    def submit_to_cluster(self, cluster_cpu, cluster_h, cluster_ram, tables, service):

        self._save_cluster_info()

        parentsample_class_pickle = os.path.join(self.cluster_dir, 'parentsample_class.pkl')
        logger.debug(f"pickling parent sample class to {parentsample_class_pickle}")
        with open(parentsample_class_pickle, "wb") as f:
            pickle.dump(self.parent_sample_class, f)

        submit_cmd = 'qsub '
        if cluster_cpu > 1:
            submit_cmd += "-pe multicore {0} -R y ".format(cluster_cpu)
        submit_cmd += f'-N wise_lightcurves '
        submit_cmd += f"-t 1-{self.n_chunks*self.n_cluster_jobs_per_chunk}:1 {self.submit_file}"
        logger.debug(f"Ram per core: {cluster_ram}")
        logger.info(f"{time.asctime(time.localtime())}: {submit_cmd}")

        self._make_cluster_script(cluster_h, cluster_ram, tables, service)

        # process = subprocess.Popen(submit_cmd, stdout=subprocess.PIPE, shell=True)
        # msg = process.stdout.read().decode()
        # logger.info(str(msg))
        # self.job_id = int(str(msg).split('job-array')[1].split('.')[0])
        # logger.info(f"Running on cluster with ID {self.job_id}")

    def run_cluster(self, cluster_cpu, cluster_h, cluster_ram, service):
        self.clear_cluster_log_dir()
        self.submit_to_cluster(cluster_cpu, cluster_h, cluster_ram, tables=None, service=service)
        self.wait_for_job()
        for c in range(self.n_chunks):
            self._combine_lcs(service, chunk_number=c, remove=True)

    # ---------------------------------------------------- #
    # END using cluster for downloading and binning        #
    # ----------------------------------------------------------------------------------- #

    #################################
    # END GET PHOTOMETRY DATA       #
    ###########################################################################################################

    ###########################################################################################################
    # START MAKE PLOTTING FUNCTIONS     #
    #####################################

    def plot_lc(self, parent_sample_idx=None, wise_id=None, interactive=False, fn=None, ax=None, save=True,
                plot_unbinned=False, plot_binned=True, lum_key='flux', service='tap', **kwargs):

        logger.debug(f"loading binned lightcurves")
        lcs = self.load_binned_lcs(service)
        unbinned_lc = None
        _get_unbinned_lcs_fct = self._get_unbinned_lightcurves if service == 'tap' else self._get_unbinned_lightcurves_gator

        if not wise_id and (parent_sample_idx is not None):
            wise_id = int(self.parent_sample.df.loc[int(parent_sample_idx), self.parent_wise_source_id_key])
            logger.debug(f"{wise_id} for {parent_sample_idx}")

        if (wise_id is not None) and not parent_sample_idx:
            m = self.parent_sample.df[self.parent_wise_source_id_key] == wise_id
            parent_sample_idx = self.parent_sample.df.index[m]
            logger.debug(f"{parent_sample_idx} for {wise_id}")

        lc = pd.DataFrame.from_dict(lcs[f"{int(parent_sample_idx)}_{wise_id}"])

        if plot_unbinned:
            _chunk_number = self._get_chunk_number(parent_sample_index=parent_sample_idx)

            if service == 'tap':
                unbinned_lcs = self._get_unbinned_lightcurves(_chunk_number)
                unbinned_lc = unbinned_lcs[unbinned_lcs.wise_id == int(wise_id)]

            else:
                unbinned_lcs = self._get_unbinned_lightcurves_gator(_chunk_number)
                unbinned_lc = unbinned_lcs[unbinned_lcs.index_01 == int(parent_sample_idx)]

        if not ax:
            fig, ax = plt.subplots(**kwargs)
        else:
            fig = plt.gcf()

        for b in self.bands:
            try:
                if plot_binned:
                    ul_mask = np.array(lc[f"{b}_{lum_key}{self.upper_limit_key}"]).astype(bool)
                    ax.errorbar(lc.mean_mjd[~ul_mask], lc[f"{b}{self.mean_key}_{lum_key}"][~ul_mask],
                                yerr=lc[f"{b}_{lum_key}{self.rms_key}"][~ul_mask],
                                label=b, ls='', marker='s', c=self.band_plot_colors[b], markersize=4,
                                markeredgecolor='k', ecolor='k', capsize=2)
                    ax.scatter(lc.mean_mjd[ul_mask], lc[f"{b}{self.mean_key}_{lum_key}"][ul_mask],
                               marker='v', c=self.band_plot_colors[b], alpha=0.7, s=2)
                if plot_unbinned:
                    ax.errorbar(unbinned_lc.mjd, unbinned_lc[f"{b}_{lum_key}"],
                                yerr=unbinned_lc[f"{b}_{lum_key}{self.error_key_ext}"],
                                label=f"{b} unbinned", ls='', marker='o', c=self.band_plot_colors[b], markersize=4,
                                alpha=0.3)
            except KeyError as e:
                logger.warning(e)

        if lum_key == 'mag':
            ylim = ax.get_ylim()
            ax.set_ylim([ylim[-1], ylim[0]])

        ax.set_xlabel('MJD')
        ax.set_ylabel(lum_key)
        ax.legend()

        if save:
            if not fn:
                fn = os.path.join(self.plots_dir, f"{wise_id}.pdf")
            logger.debug(f"saving under {fn}")
            fig.savefig(fn)

        if interactive:
            return fig, ax
        else:
            plt.close()

    #####################################
    #  END MAKE PLOTTING FUNCTIONS      #
    ###########################################################################################################

    ###########################################################################################################
    #  START CALCULATE METADATA         #
    #####################################

    def _metadata_filename(self, service, chunk_number=None, jobID=None):
        if (chunk_number is None) and (jobID is None):
            return os.path.join(self.lightcurve_dir, f'metadata_{service}.json')
        elif (chunk_number is not None) and (jobID is None):
            return os.path.join(self.cache_dir, f'metadata_{service}{self._split_chunk_key}{chunk_number}.json')
        elif (chunk_number is not None) and (jobID is not None):
            return os.path.join(self.cache_dir, f'metadata_{service}{self._split_chunk_key}{chunk_number}_job{jobID}.json')
        else:
            raise NotImplementedError

    def _load_metadata(self, service, chunk_number=None, jobID=None, remove=False):
        fn = self._metadata_filename(service, chunk_number, jobID)
        try:
            logger.debug(f"loading {fn}")
            with open(fn, "r") as f:
                metadata = json.load(f)
            if remove:
                logger.debug(f"removing")
                os.remove(fn)
            return metadata
        except FileNotFoundError:
            logger.warning(f"No file {fn}")

    def _save_metadata(self, metadata, service, chunk_number=None, jobID=None, overwrite=False):
        fn = self._metadata_filename(service, chunk_number, jobID)

        if not overwrite:
            try:
                old_metadata = self._load_metadata(service=service, chunk_number=chunk_number, jobID=jobID)
                logger.debug(f"Found {len(old_metadata)}. Combining")
                metadata = metadata.update(old_metadata)
            except FileNotFoundError as e:
                logger.info(f"FileNotFoundError: {e}. Making new metadata.")

        logger.debug(f'saving under {fn}')
        with open(fn, "w") as f:
            json.dump(metadata, f)

    def load_metadata(self, service):
        return self._load_metadata(service)

    def calculate_metadata(self, service, chunk_number=None, jobID=None, overwrite=True):
        lcs = self._load_lightcurves(service, chunk_number, jobID)
        metadata = self._calculate_metadata(lcs)
        self._save_metadata(metadata, service, chunk_number, jobID, overwrite=overwrite)

    def _combine_metadata(self, service=None, chunk_number=None, remove=False, overwrite=False):
        if not service:
            logger.info("Combining all metadata collected with all services")
            itr = ['service', ['gator', 'tap']]
            kwargs = {}
        elif chunk_number is None:
            logger.info(f"Combining all metadata collected with {service}")
            itr = ['chunk_number', range(self.n_chunks)]
            kwargs = {'service': service}
        elif chunk_number is not None:
            logger.info(f"Combining all metadata collected with {service} for chunk {chunk_number}")
            itr = ['jobID',
                   list(self.clusterJob_chunk_map.index[self.clusterJob_chunk_map.chunk_number == chunk_number])]
            kwargs = {'service': service, 'chunk_number': chunk_number}
        else:
            raise NotImplementedError

        lcs = None
        for i in itr[1]:
            kw = dict(kwargs)
            kw[itr[0]] = i
            kw['remove'] = remove
            ilcs = self._load_metadata(**kw)
            if isinstance(lcs, type(None)):
                lcs = dict(ilcs)
            else:
                lcs.update(ilcs)

        self._save_metadata(lcs, service=service, chunk_number=chunk_number, overwrite=overwrite)

    def _calculate_metadata(self, lcs):
        metadata = dict()

        for ID, lc_dict in tqdm.tqdm(lcs.items(), desc='calculating metadata', total=len(lcs)):
            imetadata = dict()
            lc = pd.DataFrame.from_dict(lc_dict)
            for band in self.bands:
                for lum_key in [self.mag_key_ext, self.flux_key_ext]:
                    llumkey = f"{band}{self.mean_key}{lum_key}"
                    difk = f"{band}_max_dif{lum_key}"
                    Nk = f"{band}_N_datapoints{lum_key}"
                    dtk = f"{band}_max_deltat{lum_key}"
                    ul_key = f'{band}{lum_key}{self.upper_limit_key}'
                    try:
                        ilc = lc[~np.array(lc[ul_key]).astype(bool)]
                        imetadata[Nk] = len(ilc)

                        if len(ilc) > 0:
                            imin = ilc[llumkey].min()
                            imax = ilc[llumkey].max()

                            if lum_key == self.mag_key_ext:
                                imetadata[difk] = imax - imin
                            else:
                                imetadata[difk] = imax / imin

                            if len(ilc) == 1:
                                imetadata[dtk] = 0
                            else:
                                mjds = np.array(ilc.mean_mjd).astype(float)
                                dt = mjds[1:] - mjds[:-1]
                                imetadata[dtk] = max(dt)

                        else:
                            imetadata[difk] = np.nan
                            imetadata[dtk] = np.nan
                    except KeyError:
                        pass

            metadata[ID] = imetadata
        return metadata

    #####################################
    #  END CALCULATE METADATA           #
    ###########################################################################################################


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--logging_level', type=str, default='INFO')
    parser.add_argument('--phot', type=bool, default=False, nargs='?', const=True)
    parser.add_argument('--perc', type=float, default=1)
    parser.add_argument('--wait', type=float, default=5)
    parser.add_argument('--service', type=str, default='tap')

    cfg = parser.parse_args()

    main_logger.setLevel(cfg.logging_level)
    logger.debug(cfg)

    start_time = time.time()
    wise_data = WISEData()
    load_time = time.time()
    #wise_data.match_all_chunks()
    match_time = time.time()
    if cfg.phot:
        wise_data.get_photometric_data(tables='NEOWISE-R Single Exposure (L1b) Source Table',
                                       perc=cfg.perc, wait=10, service='tap', mag=True, flux=False,
                                       nthreads=100, cluster_jobs_per_chunk=500)
    phot_time = time.time()

    diag_txt = f"Took {phot_time-start_time}s in total\n" \
               f"  {load_time-start_time}s for loading\n" \
               f"  {match_time-load_time}s for matching\n" \
               f"  {phot_time-match_time}s for photometry\n" \
               f"arguments: {cfg}"

    logger.info(diag_txt)

    fn = os.path.join(wise_data.cache_dir, f"{cfg.perc:.2f}_{cfg.service}_diagnostic.txt")
    with open(fn, "w") as f:
        f.write(diag_txt)
    logger.info(f"wrote to {fn}")