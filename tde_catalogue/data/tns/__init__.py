import os
import pandas as pd

from tde_catalogue import main_logger
from tde_catalogue.catalogue import Catalogue


logger = main_logger.getChild(__name__)
raw_data = f'{os.path.dirname(__file__)}/tns_search.csv'


@Catalogue.register_catalogue('TNS_catalogue')
class TNSCatalogue(Catalogue):

    def __init__(self):

        catalogue = pd.read_csv(raw_data)
        super().__init__(catalogue, f'TNS_catalogue')

    @staticmethod
    def reformat_catalogue(unformated_catalogue):
        """As the TNS format is used as the Catalogue format we can just return the catalogue"""
        formatted_catalogue = pd.DataFrame(columns=Catalogue.columns)
        for c in Catalogue.columns:
            formatted_catalogue[c] = unformated_catalogue[c]
        logger.info(f'TNS catalogue has {len(formatted_catalogue.columns)} columns')
        return formatted_catalogue