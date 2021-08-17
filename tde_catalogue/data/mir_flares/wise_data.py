import os, subprocess, copy
import pandas as pd
import numpy as np
import pyvo as vo
from tqdm import tqdm
import astropy.units as u
from astropy.table import Table

from tde_catalogue import main_logger, cache_dir, plots_dir
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

    full_cat_select = """
    SELECT
        t.source_id, t.ra, t.dec, t.sigra, t.sigdec, t.cntr
    """

    where = """
    WHERE
        t.nb<2 and 
        t.na<1 and 
        t.cc_flags like '00%'"""

    parent_sample_default_keymap = {
        'dec': 'decMean',
        'ra': 'raMean'
    }

    data_default_keymap = {
        'id': 'cntr',
        'dec': 'dec',
        'ra': 'ra',
        'dec_error': 'sigdec',
        'ra_error': 'sigra'
    }

    def __init__(self, min_sep_arcsec=60, n_chunks=8, full_cat_select=full_cat_select, where=where, base_name=base_name,
                 parent_sample_class=PanstarrsParentSample, **kwargs):

        parent_sample = parent_sample_class()
        self.full_cat_select = full_cat_select
        self.where = where
        self.base_name = base_name
        self.min_sep = min_sep_arcsec * u.arcsec
        self.store_angles_as = 'degree'

        # set up parent sample keys
        self.parent_ra_key = parent_sample.default_keymap['ra']
        self.parent_dec_key = parent_sample.default_keymap['dec']
        self.parent_wise_source_id_key = 'WISE_id'
        self.parent_sample_wise_skysep_key = 'sep_to_WISE_source'
        # set up data keys
        self.data_id_key = kwargs.pop('data_id_key', WISEData.data_default_keymap['id'])
        self.data_ra_key = kwargs.pop('data_ra_key', WISEData.data_default_keymap['ra'])
        self.data_dec_key = kwargs.pop('data_dec_key', WISEData.data_default_keymap['dec'])
        self.data_ra_error_key = kwargs.pop('data_ra_error_key', WISEData.data_default_keymap['ra_error'])
        self.data_dec_error_key = kwargs.pop('data_dec_error_key', WISEData.data_default_keymap['dec_error'])

        # set up directories
        self.cache_dir = os.path.join(cache_dir, base_name)
        self.plots_dir = os.path.join(plots_dir, base_name)

        for d in [self.cache_dir, self.plots_dir]:
            if not os.path.isdir(d):
                os.makedirs(d)

        self.parent_sample = parent_sample
        min_dec = np.floor(min(self.parent_sample.df[self.parent_ra_key]))
        max_dec = np.ceil(max(self.parent_sample.df[self.parent_dec_key]))
        logger.info(f'Declination: ({min_dec}, {max_dec})')
        self.parent_sample.df[self.parent_wise_source_id_key] = ""
        self.parent_sample.df[self.parent_sample_wise_skysep_key] = np.inf

        sin_bounds = np.linspace(np.sin(np.radians(min_dec)), np.sin(np.radians(max_dec)), n_chunks+1, endpoint=True)
        self.dec_intervalls = np.degrees(np.arcsin(np.array([sin_bounds[:-1], sin_bounds[1:]]).T))
        logger.info(f'Declination intervalls are {self.dec_intervalls}')

    def match_all_chunks(self, **table_name):
        for i in range(len(self.dec_intervalls)):
            self.match_single_chunk(i, **table_name)

    def match_single_chunk(self, chunk_number,
                           table_name="AllWISE Source Catalog"):

        m = WISEData.table_names['nice_table_name'] == table_name
        if np.any(m):
            table_name = WISEData.table_names['table_name'][m].iloc[0]

        # select the parent sample in this declination range
        dec_intervall = self.dec_intervalls[chunk_number]
        dec_intervall_mask = (self.parent_sample.df[self.parent_dec_key] > min(dec_intervall)) & \
                             (self.parent_sample.df[self.parent_dec_key] < max(dec_intervall))

        selected_parent_sample = copy.copy(self.parent_sample.df.loc[dec_intervall_mask,
                                                                     [self.parent_ra_key, self.parent_dec_key]])
        selected_parent_sample.rename(columns={self.parent_dec_key: 'dec',
                                               self.parent_ra_key: 'ra'},
                                      inplace=True)

        # write to IPAC formatted table
        _selected_parent_sample_astrotab = Table.from_pandas(selected_parent_sample)
        _parent_sample_declination_band_file = os.path.join(self.cache_dir, f"parent_sample_chunk{chunk_number}.xml")
        _selected_parent_sample_astrotab.write(_parent_sample_declination_band_file, format='ipac')

        # use Gator to query IRSA
        # TODO: use the full_cat_select input!
        _output_file = os.path.join(self.cache_dir, f"wise_data_chunk{chunk_number}.tbl")
        submit_cmd = f'curl ' \
                     f'-o {_output_file} ' \
                     f'-F filename=@{_parent_sample_declination_band_file} ' \
                     f'-F catalog={table_name} ' \
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

    def get_photometric_data(self):

        jobs = list()

        for i, r in self.parent_sample.df.iterrows():

            cntr = r[self.parent_wise_source_id_key]

            logger.debug(f"getting photometry for {cntr}")

            ##########################################################
            #      AllWISE
            ##########################################################

            q = f"""
                SELECT
                    mjd, w1flux_ep, w1sigflux_ep, w2flux_ep, w2sigflux_ep
                FROM
                    allwise_p3as_mep
                WHERE
                    cntr={cntr}
            """

            allwise_job = WISEData.service.submit_job(q)
            allwise_job.run()

            ##########################################################
            #      NEOWISE-R
            ##########################################################

            q = f"""
            SELECT
                mjd, w1flux, w1sigflux, w2flux, w2sigflux
            FROM
                neowiser_p1bs_psd
            WHERE
                allwise_cntr={cntr}
            """

            neowiser_job = WISEData.service.submit_job(q)
            neowiser_job.run()

            jobs.append([allwise_job, neowiser_job, cntr])

        for these_jobs in tqdm(jobs, desc='collecting output '):

            these_jobs[0].wait()
            allwise_lightcurve = these_jobs[0].fetch_result().to_table().to_pandas()
            allwise_lightcurve = allwise_lightcurve.rename(
                columns={
                    'w1flux_ep': 'W1_flux',
                    'w1sigflux_ep': 'W1_flux_error',
                    'w2flux_ep': 'W2_flux',
                    'w2sigflux_ep': 'W2_flux_error'
                }
            )

            these_jobs[1].wait()
            neowiser_lightcurve = these_jobs[1].fetch_result().to_table().to_pandas()
            neowiser_lightcurve = neowiser_lightcurve.rename(
                columns={
                    'w1flux': 'W1_flux',
                    'w1sigflux': 'W1_flux_error',
                    'w2flux': 'W2_flux',
                    'w2sigflux': 'W2_flux_error'
                }
            )

            combined_lightcurve = allwise_lightcurve.append(neowiser_lightcurve)