import requests
import pandas as pd
import os
import numpy as np
from difflib import get_close_matches
#See codes at https://openprescribing.net/bnf/0404/
BNF_CODES = {
        "Antidepressants": "0403",
        "ADHD_CNS_Stimulants": "0404",
        "Hypnotics": "040101"
    }


class NHSDataFetcher_OpenPrescribing(): #NHS open prescribing api

    BASE_URL = "https://openprescribing.net/api/1.0"

    # BNF Section Codes (British National Formulary)
    # 4.3: Antidepressant drugs (Proxy for Depression/Anxiety)
    # 4.4: CNS stimulants and drugs for ADHD
    # 4.1.1: Hypnotics (Proxy for sleep issues/Burnout symptoms)


    def __init__(self, output_dir="data_cache"):

        print("Initializing NHS Data Fetcher...")
        self.deprivation_handler = DeprivationHandler()
        self.deprivation_handler.load_all_imd()


        #output dir is just a dir name to save data to
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        print("NHS Data Fetcher initialized. Data will be saved to:", self.output_dir)

    def _fetch_api(self, endpoint, params=None):
        """
        Internal helper to handle API requests with basic error handling.
        """
        url = f"{self.BASE_URL}/{endpoint}/"
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data from {url}: {e}")
            return None

    def get_spending_by_ccg(self, bnf_code, start_date=None, end_date=None):
        """
        Fetches monthly spending and items for a specific BNF code/section aggregated by CCG/ICB.
        Useful for the 15-year longitudinal trend analysis.

        bnf_code: The BNF code (e.g., '0404' for ADHD).
        start_date: Optional start date (YYYY-MM-DD).
        end_date: Optional end date (YYYY-MM-DD).
        """
        print(f"Fetching regional trend data for BNF Code: {bnf_code}...")

        # The 'spending_by_ccg' endpoint allows filtering by code
        params = {
            "code": bnf_code,
            "format": "json"
        }

        data = self._fetch_api("spending_by_ccg", params=params)

        if data:
            df = pd.DataFrame(data)

            # Convert date to datetime objects immediately for easier plotting later
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])

            # Filter by date if provided (Client-side filtering as API might return all)
            if start_date:
                df = df[df['date'] >= pd.to_datetime(start_date)]
            if end_date:
                df = df[df['date'] <= pd.to_datetime(end_date)]

            return df
        return pd.DataFrame()

    def get_practice_level_data(self, bnf_code, date):
        """
        Fetches granular practice-level data for a specific single month.

        :param bnf_code: BNF code.
        :param date: Specific date string 'YYYY-MM-DD' (e.g. '2023-01-01').
        """
        print(f"Fetching practice-level stats for {bnf_code} on {date}...")

        params = {
            "code": bnf_code,
            "date": date,
            "format": "json"
        }

        # 'spending_by_practice' returns individual practice stats
        data = self._fetch_api("spending_by_practice", params=params)

        if data:
            df = pd.DataFrame(data)
            return df
        return pd.DataFrame()

    def get_icb_mapping(self):

        #Fetches the list of ICBs (Integrated Care Boards) to map codes to Region names.

        print("Fetching ICB/CCG organization details...")
        # 'org_details' endpoint
        params = {
            "org_type": "ccg",  # OpenPrescribing still uses CCG code structure for ICBs
            "format": "json"
        }
        data = self._fetch_api("org_details", params=params)
        if data:
            return pd.DataFrame(data)
        return pd.DataFrame()

    def get_geojson(self):
        """Fetches boundaries for maps."""
        url = "https://openprescribing.net/api/1.0/org_location/?org_type=ccg"
        return requests.get(url).json()

    def save_dataset(self, df, filename):
        #Helper func to save df datasets
        path = os.path.join(self.output_dir, filename)
        df.to_csv(path, index=False)
        print(f"Data saved to {path}")

    def get_org_details(self, org_type="ccg"):
        """
        Fetches organization metadata, specifically TOTAL LIST SIZE (Population)
        AND the ONS CODE (critical for linking to Deprivation data).
        """
        print(f"Fetching {org_type} details (including population and ONS codes)...")
        # Added 'ons_code' to the keys list
        params = {
            "org_type": org_type,
            "keys": "total_list_size,name,code,ons_code,open_date,close_date",
            "format": "json"
        }
        data = self._fetch_api("org_details", params=params)
        if data:
            df = pd.DataFrame(data)
            df['total_list_size'] = pd.to_numeric(df['total_list_size'], errors='coerce')
            return df
        return pd.DataFrame()

    def merge_deprivation(self, df):
        """
        Apply the temporal IMD lookup to a prescription dataframe.
        """
        print("Merging temporal Deprivation Data (this may take a moment)...")

        # We define a helper to apply row-wise
        # Optimization: Instead of row-by-row, we can map years to IMD versions
        # But for clarity and robustness with the specific user files, let's use a function

        # Create a dictionary to store results
        records = []

        for _, row in df.iterrows():
            name = row['row_name_x']
            date = row['date']
            score = self.deprivation_handler.get_imd_score_by_name(name, date)
            # create record including merge keys so we can join back
            rec = {k: row[k] for k in ['row_name_x', 'date', "match_key"] if k in row}
            rec.update(score)
            records.append(rec)
        
        imd_df = pd.DataFrame.from_records(records)
        cols = imd_df.loc[:, 'imd_score':'idaopi_score'].columns
        imd_df[cols] = imd_df[cols].apply(pd.to_numeric, errors='coerce')
        # If no IMD columns were produced, return original df
        if imd_df.empty or imd_df.shape[1] == 2:
            # No enrichment available
            return df

        # Merge and return (left join to keep original rows)
        merged_df = pd.merge(df, imd_df, on=['row_name_x', 'date'], how='left')

        return merged_df  

    def get_national_trends(self, bnf_code):
        """
        Fetches the aggregate national spending for England over all available time.
        Useful for the 15-year longitudinal view (ignoring regional boundary changes).
        """
        print(f"Fetching National 15-year trends for {bnf_code}...")
        params = {
            "code": bnf_code,
            "format": "json"
        }
        # The 'spending' endpoint returns national totals over time
        data = self._fetch_api("spending", params=params)

        if data:
            df = pd.DataFrame(data)
            df['date'] = pd.to_datetime(df['date'])
            return df
        return pd.DataFrame()

    def get_ccg_boundaries(self):
        """
        Fetches CCG/ICB boundary data (GeoJSON) for creating choropleth maps.
        Returns GeoJSON format data for mapping.
        """
        print("Fetching CCG/ICB geographic boundaries...")
        try:
            # OpenPrescribing provides boundary data
            url = "https://openprescribing.net/api/1.0/org_location/?org_type=ccg&format=json"
            response = requests.get(url)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching boundary data: {e}")
            return None

    def get_uk_boundaries_alternative(self):
        """
        Alternative source for UK boundary data if OpenPrescribing doesn't provide sufficient detail.
        Uses ONS (Office for National Statistics) boundary data.
        """
        print("Fetching UK boundary data from ONS...")
        try:
            # ONS provides Clinical Commissioning Group boundaries
            url = "https://opendata.arcgis.com/datasets/b216b4c8a4e74f6fb692a1785255d777_0.geojson"
            response = requests.get(url)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching ONS boundary data: {e}")
            # Fallback to a simpler boundary dataset
            try:
                # Simplified UK regions as backup
                url = "https://raw.githubusercontent.com/holtzy/The-Python-Graph-Gallery/master/static/data/uk-counties.geojson"
                response = requests.get(url)
                response.raise_for_status()
                return response.json()
            except:
                print("Could not fetch any boundary data")
                return None


class DeprivationHandler:
    def __init__(self, base_path="util/source/deprivation"):
        self.base_path = base_path
        self.imd_cache = {}

    def _clean_name(self, name):
        """
        Standardises names by lowercase, stripping, and removing common suffixes.
        'NHS Leeds CCG' -> 'leeds'
        """
        if pd.isna(name):
            return ""

        name = str(name).lower().strip()

        # Remove common suffixes/prefixes to find the 'core' name
        replacements = ["nhs ", "ccg", "icb", "integrated care board", "clinical commissioning group"]
        for r in replacements:
            name = name.replace(r, "")

        return name.strip()

    def _load_excel(self, filename):
        path = os.path.join(self.base_path, filename)
        try:
            imd_scores = {}
            dimensions = pd.ExcelFile(path).sheet_names[1:]  # Skip summary sheet

            for sheet in dimensions:
                # Column 1 = Name, Column 3 = Score 
                df = pd.read_excel(path, sheet_name=sheet, usecols=[1, 2])
                sheet_clean = self._clean_name(sheet)
                df.columns = ['org_name', f'{sheet_clean}_score']
                imd_scores['org_name'] = df['org_name']
                imd_scores[f'{sheet_clean}_score'] = df[f'{sheet_clean}_score']
            df = pd.DataFrame(imd_scores)
            # Apply the cleaning function to create a matching key
            df['match_key'] = df['org_name'].apply(self._clean_name)
            return df
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            return pd.DataFrame()

    def load_all_imd(self):
        print("Loading local IMD datasets (Names & Scores)...")
        self.imd_cache['2015'] = self._load_excel("imd2015.xlsx")
        self.imd_cache['2019'] = self._load_excel("imd2019.xlsx")
        self.imd_cache['2025'] = self._load_excel("imd2025.xlsx")

    def get_imd_score_by_name(self, target_name, date_obj):
        """
        Finds IMD score using cleaned name matching with fallback.
        """
        clean_target = self._clean_name(target_name)
        #print(clean_target)

        year = date_obj.year

        # 1. Determine priority order based on date
        # If year is 2021, try 2019 first, then fallbacks.
        if year < 2019:
            priority = ['2015', '2019', '2025']
        elif year < 2025:
            priority = ['2019', '2025', '2015']
        else:
            priority = ['2025', '2019', '2015']

        # 2. Iterate through datasets in priority order
        for key in priority:
            df = self.imd_cache.get(key)
            #print(df)
            #print(df.columns) #debug
            if df is not None and not df.empty:
                match = df[df['match_key'] == clean_target]
                if not match.empty:
                    return match.iloc[0][1:]
            else:
                match = df.loc[:, 'imd_score':'idaopi_score'].mean().to_frame()  # Average across dimensions

        #no exact match found, return average
        return match

if __name__ == "__main__":
    fetcher = NHSDataFetcher_OpenPrescribing()
    x = fetcher.get_national_trends(BNF_CODES["ADHD_CNS_Stimulants"])
    print(x)