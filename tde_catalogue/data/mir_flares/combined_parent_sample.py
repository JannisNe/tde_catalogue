import os, argparse, logging
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy import units as u

from tde_catalogue import main_logger, cache_dir, plots_dir
from tde_catalogue.data.mir_flares import base_name as mir_base_name
from tde_catalogue.data.mir_flares.panstarrs_parent_sample import PanstarrsParentSample
from tde_catalogue.data.mir_flares.sdss_parnet_sample import SDSSParentSample
from tde_catalogue.data.mir_flares.parent_sample import ParentSample


logger = main_logger.getChild(__name__)


class CombinedParentSample(ParentSample):

    base_name = f"{mir_base_name}/combined_sample"
    default_keymap = {
        'ra': 'ra',
        'dec': 'dec'
    }

    def __init__(self, parent_sample_classes, min_sep=20,
                 base_name=base_name, store=True):

        assert len(parent_sample_classes) == 2

        self.parent_sample_classes = parent_sample_classes
        self.base_name = base_name
        self.min_sep = min_sep * u.arcsec
        self._store = store

        # set up directories
        self.cache_dir = os.path.join(cache_dir, base_name)
        self.plots_dir = os.path.join(plots_dir, base_name)

        for d in [self.cache_dir, self.plots_dir]:
            if not os.path.isdir(d):
                os.makedirs(d)

        if (not os.path.isfile(self.local_sample_copy)) or (not self._store):
            self._combine_samples()

        if self._store:
            logger.debug(f"loading from {self.local_sample_copy}")
            self.df = pd.read_csv(self.local_sample_copy)

    @property
    def local_sample_copy(self):
        return os.path.join(self.cache_dir, "sample.csv")

    def _combine_samples(self):
        #######################################################################################
        # START COMBINING #
        ###################

        logger.debug("initialising parent samples")
        parent_samples = [p() for p in np.atleast_1d(self.parent_sample_classes)]

        # get the parent sample DataFrames and rename the ra and dec columns
        # so they end up being what is specified in self.default_keymap
        dfs = [
            p.df.rename(columns={p.default_keymap[k]: v for k, v in self.default_keymap.items()})
            for p in parent_samples
        ]

        logger.debug(f"initialising SkyCoord's")
        skycoords = [
            SkyCoord(df[self.default_keymap['ra']], df[self.default_keymap['dec']],
                     unit='deg')
            for df in dfs
        ]

        # match smaller catalog to bigger one
        lengths = [len(df) for df in dfs]
        sorted_length = np.argsort(lengths)
        logger.info('matching catalogs ...')
        i, sep, _ = skycoords[sorted_length[0]].match_to_catalog_sky(skycoords[sorted_length[1]])

        # consider matches where sources are closer than min_sep together
        idf = pd.DataFrame({'i': i, 'sep': sep})
        accept_sep = idf.sep <= self.min_sep.to('deg')
        logger.info(f"{len(idf[accept_sep])} matches with small enough sky separation.")

        # for duplicated matches, chose the one with smaller skysep
        idf_sorted_sep = idf.sort_values('sep')
        idf_sorted_sep['accept_dup'] = ~idf_sorted_sep.i.duplicated(keep='first')
        idf = idf_sorted_sep.sort_index()
        idf['accept_m'] = idf['accept_dup'] & accept_sep

        logger.info(f"found multiple matches for {len(idf[~idf.accept_dup])} objects")
        logger.info(f"Of {len(idf)} matches kept {len(idf[idf.accept_m])}. "
                    f"Declined {len(idf) - len(idf[idf.accept_m])}")

        accept_m = np.array(idf.accept_m)
        n_matches = len(dfs[sorted_length[0]][accept_m])
        combined_cat_len = sum(lengths) - n_matches
        logger.info(f"Found {n_matches} matches. Combined catalog will have {combined_cat_len} objects.")

        # re-index the smaller catalog
        # create an empty index array for the smaller catalog
        new_index = np.empty(lengths[sorted_length[0]])
        # the index of the accepted matches are the index of the match
        new_index[accept_m] = i[accept_m]
        # the declined objects will be appended to the larger catalog
        # their indices start after the last index of the larger catalog
        index_start = lengths[sorted_length[1]] + 1
        index_end = index_start + lengths[sorted_length[0]] - n_matches
        new_index[~accept_m] = list(range(index_start, index_end))
        # set the new index to a copy of the small catalog
        new_small_cat = dfs[sorted_length[0]]
        new_small_cat.index = new_index.astype(int)
        # concatenate both catalogs
        big_cat = dfs[sorted_length[1]].copy()

        # rename the columns so it gets clear where they come from
        new_small_cat = new_small_cat.rename(
            columns={
                k: f"{parent_samples[sorted_length[0]].base_name}_{k}"
                for k in new_small_cat.columns
            }
        )

        big_cat = big_cat.rename(
            columns={
                k: f"{parent_samples[sorted_length[1]].base_name}_{k}"
                for k in big_cat.columns
            }
        )

        combined_cat = pd.concat([big_cat, new_small_cat], axis=1)
        # ensure that the length is correct
        if not len(combined_cat) == combined_cat_len:
            raise Exception
        # set ra and dec column
        for k, v in self.default_keymap.items():
            # use the values from the small cat where possible
            small_cat_k = f"{parent_samples[sorted_length[0]].base_name}_{v}"
            combined_cat[v] = combined_cat[small_cat_k]
            # where values are missing insert from big cat
            big_cat_k = f"{parent_samples[sorted_length[1]].base_name}_{v}"
            missing_values_m = combined_cat[v].isna()
            combined_cat.loc[missing_values_m, v] = combined_cat[big_cat_k][missing_values_m]

        self.df = combined_cat

        # clear parent_samples for saving memory
        del parent_samples

        ###################
        # END COMBINING   #
        #######################################################################################

        if self._store:
            logger.debug(f'saving to {self.local_sample_copy}')
            self.df.to_csv(self.local_sample_copy)

    def plot_cutout(self, *args, **kwargs):
        res = list()
        for p in np.atleast_1d(self.parent_sample_classes):
            inst = p()
            r = inst.plot_cutout(*args, **kwargs)
            if not isinstance(r, type(None)):
                res.append(r)
        if len(res) > 0:
            return res


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--logging_level', type=str, default='INFO')
    cfg = parser.parse_args()

    main_logger.setLevel(cfg.logging_level)

    CombinedParentSample([SDSSParentSample, PanstarrsParentSample])