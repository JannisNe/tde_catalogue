import os
import pandas as pd
import numpy as np
import pyvo as vo
from tqdm import tqdm
from astropy.coordinates import SkyCoord, Angle
import astropy.units as u

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
        t.source_id, t.ra, t.dec, t.sigra, t.sigdec
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
        'id': 'source_id',
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
        min_dec = np.floor(min(self.parent_sample.df.decMean))
        max_dec = np.ceil(max(self.parent_sample.df.decMean))
        logger.info(f'Declination: ({min_dec}, {max_dec})')
        self.parent_sample.df[self.parent_wise_source_id_key] = ""
        self.parent_sample.df[self.parent_sample_wise_skysep_key] = np.inf

        sin_bounds = np.linspace(np.sin(np.radians(min_dec)), np.sin(np.radians(max_dec)), n_chunks)
        self.dec_intervalls = np.degrees(np.arcsin(np.array([sin_bounds[:-1], sin_bounds[1:]]).T))
        logger.info(f'Declination intervalls are {self.dec_intervalls}')

    def match_all_chunks(self, table_name):
        for i in range(len(self.dec_intervalls)):
            self.match_single_chunk(i, table_name)

    def get_tap_output(self, chunk_number, table_name):
        dec_intervall = self.dec_intervalls[chunk_number]
        where = f"""{self.where} and
                    t.dec>{dec_intervall[0]} and
                    t.dec<{dec_intervall[1]}
                """

        queue = f"""
                {self.full_cat_select}
                FROM
                    {table_name} t
                {where}
                """

        logger.info(f"Queue: {queue}")
        query_job = WISEData.service.submit_job(queue)
        query_job.run()
        logger.info(f'Job: {query_job.url}; {query_job.phase}')
        logger.info('waiting ...')
        query_job.wait()
        logger.info('Done!')

        tap_res = query_job.fetch_result().to_table().to_pandas()
        return tap_res

    def match_single_chunk(self, chunk_number, table_name):

        tap_res = self.get_tap_output(chunk_number, table_name)

        logger.info(f'matching {len(tap_res)} objects from query result to '
                    f'{len(self.parent_sample.df)} objects from parent sample')

        if len(tap_res) == 0:
            logger.warning(f'No objects to match to! Skipping!')

        else:
            parent_dec = self.parent_sample.df[self.parent_dec_key]
            parent_ra = self.parent_sample.df[self.parent_ra_key]

            dd, dde = tap_res[self.data_dec_key], tap_res[self.data_dec_error_key]
            dd_minus_dde = dd - dde
            dd_plus_dde = dd + dde

            chunk_mask = (parent_dec >= min(dd_minus_dde)) & (parent_dec <= max(dd_plus_dde))
            chunk_indices = np.where(chunk_mask)[0]
            logger.debug(f'{len(parent_ra[chunk_mask])} in this chunk')

            logger.debug('doing the catalogue matching ...')
            parent_sample_coord = SkyCoord(parent_ra[chunk_mask] * u.degree,
                                           parent_dec[chunk_mask] * u.degree)
            query_tap_result_coord = SkyCoord(tap_res[self.data_ra_key] * u.degree,
                                              tap_res[self.data_dec_key] * u.degree)

            index, sky_sep, _ = parent_sample_coord.match_to_catalog_sky(query_tap_result_coord)
            logger.debug(f'Index: {index[:10]} ...')
            logger.debug(f'skyse: {sky_sep[:10]} ...')

            logger.debug(f'shape of index is {np.shape(index)}')
            new_closest_source_mask = sky_sep.to(self.store_angles_as).value < self.parent_sample.df[self.parent_sample_wise_skysep_key][chunk_mask]
            new_closest_source_index = index[new_closest_source_mask]
            logger.debug(f'shape of new source mask is {np.shape(new_closest_source_mask)}')
            source_ids = tap_res.iloc[new_closest_source_index][self.data_id_key]
            source_skysep = sky_sep[new_closest_source_mask].to(self.store_angles_as).value

            self.parent_sample.df.loc[
                chunk_indices[new_closest_source_mask],
                self.parent_sample_wise_skysep_key
            ] = list(source_skysep)

            self.parent_sample.df.loc[
                chunk_indices[new_closest_source_mask],
                self.parent_wise_source_id_key
            ] = list(source_ids)
