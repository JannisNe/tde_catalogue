import abc, os
import pandas as pd
import numpy as np

from tde_catalogue import main_logger, cache_dir, plots_dir


logger = main_logger.getChild(__name__)


class ParentSample(abc.ABC):

    df = pd.DataFrame
    default_keymap = dict

    def __init__(self, base_name):
        # set up directories
        self.cache_dir = os.path.join(cache_dir, base_name)
        self.plots_dir = os.path.join(plots_dir, base_name)

        for d in [self.cache_dir, self.plots_dir]:
            if not os.path.isdir(d):
                os.makedirs(d)

    def plot_cutout(self, ind, arcsec=20, interactive=False, **kwargs):
        """
        Plot the coutout images in all filters around the position of object with index i
        """
        sel = self.df.iloc[np.atleast_1d(ind)]
        ra, dec = sel[self.default_keymap["ra"]], sel[self.default_keymap["dec"]]
        title = [r[self.default_keymap["id"]] for i, r in sel.iterrows()]
        fn = kwargs.pop("fn", [f"{i}_{r[self.default_keymap['id']]}.pdf" for i, r in sel.iterrows()])
        logger.debug(f"\nRA: {ra}\nDEC: {dec}\nTITLE: {title}\nFN: {fn}")
        ou = list()

        ras = np.atleast_1d(ra)
        decs = np.atleast_1d(dec)
        title = np.atleast_1d(title) if title else [None] * len(ras)
        fn = np.atleast_1d(fn) if fn else [None] * len(ras)

        for _ra, _dec, _title, _fn in zip(ras, decs, title, fn):
            ou.append(self._plot_cutout(_ra, _dec, arcsec, interactive, title=_title, fn=_fn, **kwargs))

        if len(ou) == 1:
            ou = ou[0]
        return ou

    @abc.abstractmethod
    def _plot_cutout(self, ra, dec, arcsec, interactive, title=None, fn=None, save=False, ax=None, **kwargs):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def local_sample_copy(self):
        raise NotImplementedError

    def save_local(self):
        logger.debug(f"saving under {self.local_sample_copy}")
        self.df.to_csv(self.local_sample_copy)