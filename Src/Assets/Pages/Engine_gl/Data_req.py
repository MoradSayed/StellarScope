import requests, pandas as pd, os, json, timeit
from io import StringIO
import re

def get_exoplanet_data(table, columns):
    # Set the base URL for the Exoplanet Archive API
    url = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"

    # Define the SQL query to get exoplanet data, including Right Ascension (ra) and Declination (dec)
    selectStr, whereStr = columns
    query = f"""
    SELECT {selectStr}
    FROM {table}
    {whereStr}
    """

    params = {
        "query": query,
        "format": "csv"
    }

    response = requests.get(url, params=params)

    return response.text

def csv_to_json(f1, f2):
    empty, exst = 0, 0

    df1 = pd.read_csv(StringIO(f1))
    df2 = pd.read_csv(StringIO(f2))

    df1 = df1.drop_duplicates(subset=["sy_name"])

    result = {}
    for index, row in df1.iterrows():
        system_name = row['sy_name']
        right_as = row['ra']
        dec = row['dec']
        temp = row['st_teff']
        temp = temp if not pd.isna(temp) else None
        rad = row['st_rad']
        rad = rad if not pd.isna(rad) else None
        sys_distance = row['sy_dist']
        # Filter the second dataframe for matching 'item'
        matching_rows = df2[df2['hostname'].apply(lambda x: re.match(rf"^{re.escape(system_name)}(\s[A-Z])?$", x) is not None)] # .str.startswith(system_name, na=False)]  # == system_name
        # Extract relevant columns (ignoring the 'item' column itself)
        matches = matching_rows[['pl_name', ]].apply(lambda x: ' '.join(x.astype(str)), axis=1).tolist()    #! add any planet specific data here
        result[system_name] = [right_as, dec, sys_distance, temp, rad, matches]

        if temp:
            exst += 1
        else:
            empty += 1
    print(f"empty, exst: {empty}, {exst}")

    with open(os.path.join("Src", "Assets", "Cache", 'new_combined_file.json'), 'w') as f:
        json.dump(result, f, indent=4)

def main():
    responces = []
    for table in tables.keys():
        responces.append(get_exoplanet_data(table, tables[table]))
        #* Option #1: save as csv
        with open(os.path.join("Src", "Assets", "Cache", f"{table}_info.csv"), "w") as f:
            f.write(responces[-1])
    #* Option #2: save as json    
    # print(timeit.timeit(lambda: csv_to_json(*responces), number=1))



# tables = {
#     "stellarhosts": ("sy_name, ra, dec, sy_dist, st_teff, st_rad", "WHERE sy_dist IS NOT NULL"),
#     "pscomppars": ("pl_name, ra, dec, sy_dist, hostname", "")
# }
tables = {
    "stellarhosts": ("sy_name, st_mass, st_met, st_metratio, st_lum, st_logg, st_age, st_dens", ""),
    "pscomppars": ("pl_name, discoverymethod, disc_year, disc_telescope, pl_orbper, pl_orbsmax, pl_rade, pl_radj, pl_bmasse, pl_bmassj, pl_dens, pl_eqt", "")
}

main()

# with open(os.path.join("Src", "Assets", "Cache", 'combined_file.json'), 'r') as f:
#     data = json.load(f)
#     no_of_planets = 0
#     for planets in data.values():
#         no_of_planets += len(planets[3])
#     print(no_of_planets)

