import re
import pandas as pd

class CongressCanonicalNames:
    _CURRENT_URL = "https://unitedstates.github.io/congress-legislators/legislators-current.csv"
    _HISTORICAL_URL = "https://unitedstates.github.io/congress-legislators/legislators-historical.csv"


    @staticmethod
    def _clean_name(name):
        return re.sub(r"\s*\([^()]*\)$", "", name)

    @classmethod
    def _load_dataframe(cls):
        df_current = pd.read_csv(cls._CURRENT_URL)
        df_hist = pd.read_csv(cls._HISTORICAL_URL)
        df = pd.concat([df_current, df_hist], ignore_index=True)

        df["bioname_canonical"] = df["wikipedia_id"].apply(cls._clean_name)
        return df

    @classmethod
    def get_canonical_names(cls):
        """
        Return a (bioguide_id, bioname_canonical) DataFrame
        """
        df = cls._load_dataframe()
        return df[["bioguide_id", "bioname_canonical"]].copy()