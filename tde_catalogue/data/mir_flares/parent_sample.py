import abc
import pandas as pd

from tde_catalogue import main_logger


logger = main_logger.getChild(__name__)


class ParentSample(abc.ABC):

    df = pd.DataFrame

    @abc.abstractmethod
    def plot_cutout(self, ind, arcsec=20, interactive=False, **kwargs):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def local_sample_copy(self):
        raise NotImplementedError

    def save_local(self):
        logger.debug(f"saving under {self.local_sample_copy}")
        self.df.to_csv(self.local_sample_copy)