import abc


class Catalogue(abc.ABC):

    columns = [
        "ID",
        "Name",
        "RA",
        "DEC",
        "Obj. Type",
        "Redshift",
        "Host Name",
        "Host Redshift",
        "Reporting Group/s",
        "Associated Group/s",
        "Discovery Data Source/s",
        "Classifying Group/s",
        "Disc. Internal Name",
        "Disc. Instrument/s",
        "Class. Instrument/s",
        "Public",
        "Discovery Mag/Flux",
        "Discovery Filter",
        "Discovery Date (UT)",
        "Sender",
        "Ext. catalog/s",
        "Remarks"
    ]

    registered_catalogues = dict()

    def __init__(self, catalogue, name):
        """
        Base Catalogue class
        :param catalogue: pandas.Dataframe, containing the TDE catalogue
        :param name: str, the name of the catalogue
        """

        self.raw_catalogue = catalogue
        self.catalogue = self.reformat_catalogue(catalogue)
        self.name = name

    def __getitem__(self, item):
        return self.catalogue[item]

    @staticmethod
    @abc.abstractmethod
    def reformat_catalogue(unformated_catalogue):
        """Takes the raw catalogue and returns it in the uniform data format"""
        raise NotImplementedError

    @classmethod
    def register_catalogue(cls, catalogue_name):
        """
        Adds a new subclass of Catalogue
        """
        def decorator(subclass):
            cls.registered_catalogues[catalogue_name] = subclass
            return subclass

        return decorator