# get census data using census package
# adapted from ISYE6740 project

from census import Census
import pandas as pd
import config
import re

key = config.acs_key

c = Census(key)

def import_codes(filename):

    acs_codes = open(filename)
    codes=acs_codes.read()
    acs_codes.close()
    codes = ",".join(codes.splitlines())
    vars = codes.split(',')

    vars.insert(0,"NAME")

    return vars

namelist = import_codes('acs_variables.csv') #one per line, no commas

va_census = c.acs5.state_county_tract(fields = namelist,
                                      state_fips = "51",
                                      county_fips = "*",
                                      tract = "*",
                                      year = 2018)
# Create a dataframe from the census data
va_df = pd.DataFrame(va_census)

# Show the dataframe
print(va_df.head(2))
print('Shape: ', va_df.shape)

numeric_const_pattern = '[-+]? (?: (?: \d* \. \d+ ) | (?: \d+ \.? ) )(?: [Ee] [+-]? \d+ ) ?'
rx = re.compile(numeric_const_pattern, re.VERBOSE)
tractnames = va_df['NAME'].to_list()

tractnumbers = []
for name in tractnames:
    tractnumbers.append(rx.findall(name)[0])

va_df['number'] = tractnumbers
geoid_list = va_df["GEO_ID"].tolist()
geoid = [x[9:] for x in geoid_list]
va_df["GEOID"] = geoid
va_df.to_csv('acs_data.csv',index=False)