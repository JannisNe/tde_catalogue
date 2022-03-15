import os, logging
import numpy as np

from timewise import WISEDataDESYCluster
from timewise.general import main_logger


logger = logging.getLogger(__name__)


class WISEDataWithKnownDesignation(WISEDataDESYCluster):

    def _match_single_chunk(self, chunk_number, table_name):
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

        gatorres_df = gator_res.to_pandas(index='index_01')

        all_good_mask = self.parent_sample.df[dec_intervall_mask].AllWISE_designation == gatorres_df.designation

        if not np.all(all_good_mask):
            logger.info('Some designations do not match in the GATOR query result and the NEWS table')
            logger.info(self.parent_sample.df[dec_intervall_mask][~all_good_mask].to_string(
                columns=['RAdeg', 'DEdeg', 'AllWISE_designation']))
            logger.info(gatorres_df[~all_good_mask].to_string(columns=['ra', 'dec', 'designation']))

            for i in (all_good_mask.index[~all_good_mask]):
                iy = self.parent_sample.df[dec_intervall_mask].AllWISE_designation.loc[i]
                iz = gatorres_df.designation.loc[i]

                logger.debug(f"NEWS designation: {iy}, GATOR designation: {iz}")
                if iy.endswith('0') and iz.endswith('0'):
                    same = False
                    if '-' in iy:
                        same = iy.replace('-', '+') == iz
                    if '+' in iy:
                        same = iy.replace('+', '-') == iz

                    if same:
                        logger.debug("are the same")
                        all_good_mask.loc[i] = True
                    else:
                        raise Exception

        self.parent_sample.df.loc[gatorres_df.index, 'AllWISE_id'] = gatorres_df.cntr