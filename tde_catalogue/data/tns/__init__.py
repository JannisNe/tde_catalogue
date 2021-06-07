import os
import pandas as pd

from tde_catalogue.catalogue import Catalogue


raw_data = f'{os.path.dirname(__file__)}/tns_search.csv'


class TNSCatalogue(Catalogue):

    def __init__(self):

        catalogue = pd.read_csv(raw_data)
        super().__init__(catalogue, f'TNS_catalogue')