import astropy.units as u
from astropy.coordinates import SkyCoord
import pandas as pd
import numpy as np
import tqdm

from tde_catalogue import main_logger
from tde_catalogue.catalogue import Catalogue


logger = main_logger.getChild(__name__)


class Compiler:
    """
    An instance of this class will take a list of catalogues and
    merge them into one
    """

    columns = Catalogue.columns

    def __init__(self, catalogues, check_angular_distance_arcs=0.1*u.degree, check_time_d=np.inf,
                 ra_key='RA', dec_key='DEC', time_key='Discovery Date (UT)',
                 ra_unit=u.hourangle, dec_unit=u.degree):
        """

        :param catalogues: list of Catalogue instances
        :param check_angular_distance_arcs:
        :param check_time_d:
        :param ra_key:
        :param dec_key:
        :param time_key:
        :param ra_unit:
        :param dec_unit:
        """

        # assert that all catalogues have the requested columns
        for cat in catalogues:
            assert isinstance(cat, Catalogue)
            assert len(cat.catalogue.columns) == len(Compiler.columns)
            assert np.all(cat.catalogue.columns == Compiler.columns)

        # concatenate catalogues
        dfs = [cat.catalogue for cat in catalogues]
        self.concatenated_cat = pd.concat(dfs)

        # set quantities that are used to check for duplicates
        self.ra_key = ra_key
        self.dec_key = dec_key
        self.ra_unit = ra_unit
        self.dec_unit=dec_unit
        self.time_key = time_key
        self.check_angular_distance_arcs = check_angular_distance_arcs
        self.check_time_d = check_time_d
        self.check_for_duplicates()

    def check_for_duplicates(self):
        """Raises a compiler error when two events are found at the same position and time"""
        for i, row1 in tqdm.tqdm(self.concatenated_cat.iterrows(), desc='checking for duplicates',
                                 total=len(self.concatenated_cat)):
            coordinate1 = SkyCoord(row1[self.ra_key], row1[self.dec_key],
                                   unit=(self.ra_unit, self.dec_unit))

            for j in range(int(i) + 1, len(self.concatenated_cat)):
                row2 = self.concatenated_cat.iloc[j]
                coordinate2 = SkyCoord(row2[self.ra_key], row2[self.dec_key],
                                       unit=(self.ra_unit, self.dec_unit))

                ang_sep = coordinate1.separation(coordinate2)
                if ang_sep < self.check_angular_distance_arcs:
                    raise CompilerError(f'\n Found duplicates in the catalogue!:\n'
                                        f'{row1} \n{row2}')

        logger.info('No duplicates found.')


class CompilerError(Exception):
    pass
