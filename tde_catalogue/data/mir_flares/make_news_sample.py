import os, logging
import numpy as np

from timewise import ParentSampleBase, WiseDataByVisit, WISEDataDESYCluster
from timewise.general import data_dir, main_logger

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
        n_chunks=320,
        min_sep_arcsec=6
    )

    logger.info('getting AllWISE cntr')
    wd.match_all_chunks()

    # table_name="AllWISE Source Catalog"
    #
    # try:
    #     for chunk_number in range(29, wd.n_chunks):
    #         dec_intervall_mask = wd.chunk_map == chunk_number
    #         logger.debug(f"Any selected: {np.any(dec_intervall_mask)}")
    #         _parent_sample_declination_band_file = os.path.join(wd.cache_dir, f"parent_sample_chunk{chunk_number}.xml")
    #         _output_file = os.path.join(wd.cache_dir, f"parent_sample_chunk{chunk_number}.tbl")
    #         gator_res = wd._match_to_wise(
    #             in_filename=_parent_sample_declination_band_file,
    #             out_filename=_output_file,
    #             mask=dec_intervall_mask,
    #             table_name=table_name
    #         )
    #
    #         for fn in [_parent_sample_declination_band_file, _output_file]:
    #             try:
    #                 logger.debug(f"removing {fn}")
    #                 os.remove(fn)
    #             except FileNotFoundError:
    #                 logger.warning(f"No File!!")
    #
    #         gatorres_df = gator_res.to_pandas(index='index_01')
    #
    #         all_good_mask = wd.parent_sample.df[dec_intervall_mask].AllWISE_designation == gatorres_df.designation
    #
    #         if not np.all(all_good_mask):
    #             logger.info('Some designations do not match in the GATOR query result and the NEWS table')
    #             logger.info(wd.parent_sample.df[dec_intervall_mask][~all_good_mask].to_string(columns=['RAdeg', 'DEdeg','AllWISE_designation']))
    #             logger.info(gatorres_df[~all_good_mask].to_string(columns=['ra', 'dec', 'designation']))
    #
    #             for i in (all_good_mask.index[~all_good_mask]):
    #                 iy = wd.parent_sample.df[dec_intervall_mask].AllWISE_designation.loc[i]
    #                 iz = gatorres_df.designation.loc[i]
    #
    #                 logger.debug(f"NEWS designation: {iy}, GATOR designation: {iz}")
    #                 if iy.endswith('0') and iz.endswith('0'):
    #                     same = False
    #                     if '-' in iy:
    #                         same = iy.replace('-', '+') == iz
    #                     if '+' in iy:
    #                         same = iy.replace('+', '-') == iz
    #
    #                     if same:
    #                         logger.debug("are the same")
    #                         all_good_mask.loc[i] = True
    #                     else:
    #                         raise Exception
    #
    #         wd.parent_sample.df.loc[gatorres_df.index, 'AllWISE_id'] = gatorres_df.cntr.astype(str)
    #
    # finally:
    #     logger.info(f"saving under {wd.parent_sample.local_sample_copy}")
    #     wd.parent_sample.df.to_csv(wd.parent_sample.local_sample_copy)