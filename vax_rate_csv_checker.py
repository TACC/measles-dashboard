# This script checks states' vaccination rate CSVs for validity and
#   proper formatting -- this data input pipeline is very important
#   to efficiently and responsibly add new states to the dashboard.
# Vaccination rate CSVs must follow a very specific format
#   to work with the dashboard.
# The dashboard itself should not have to consider or adapt to
#   all the possible formatting options -- this should be
#   handled in the data pipeline, not the dashboard.

# To use, run the function check_vaccination_rate_CSV on
#   the filename of the CSV file that must be validated.

# import sys
# pd.set_option('display.precision', 1)
# np.set_printoptions(precision=4)
# np.set_printoptions(linewidth=185)
# pd.set_option('display.width', 185)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
# pd.options.display.float_format = '{:,.4}'.format
# np.set_printoptions(threshold=sys.maxsize)

# %% Imports
import os
import sys
import numpy as np
import pandas as pd

# %% Print options
pd.set_option('display.precision', 1)
pd.set_option('display.width', 185)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.options.display.float_format = '{:,.4}'.format

np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(precision=4)
np.set_printoptions(linewidth=185)

# %% Functions
COLUMNS_LIST = ['Facility Number', 'School Type', 'School District or Name',
                'Facility Address', 'County', 'MMR Vaccination Rate', 'Age Group']


def check_columns_list(df):
    if not set(df.columns) == set(COLUMNS_LIST):
        print("\n\nError: CSV columns do not match COLUMNS_LIST. "
              "Make sure all columns in COLUMNS_LIST exist in the CSV"
              " and spelling and formatting are correct.")
        
        print('\nColumns in csv file:')
        print(df.columns)
        print('\nColumns expected:')
        print(COLUMNS_LIST)
        print('\nColumns missing:')
        print([x for x in COLUMNS_LIST if x not in df.columns])
        print('\nUnnecessary columns:')
        print([x for x in df.columns if x not in COLUMNS_LIST])
        
        return False

    else:
        return True


def check_no_duplicates(df):
    # Important note: some schools in different counties have the same name!
    #   A bit tricky!
    
    groupby_cols = [
        "School District or Name", "Age Group", "County"
        ]
    df_agg = df.groupby(groupby_cols).nunique()        
    
    if len(df) != len(df_agg):
        print("\n\nError: Combinations created by values of School District or Name, "
              "Age Group, and County must be unique. There is at least one "
              "duplicate -- please revise.")
        df_agg['duplicate_row_max'] = df_agg.apply(lambda x: max(x), axis=1)
        df_issues = df_agg.loc[df_agg['duplicate_row_max'] > 1]
        print('\nValues causing issues:')
        print(df_issues)
        print('\n\n')
        
        return False
    else:
        return True


def check_vaccination_rates_floats(df):
    if df["MMR Vaccination Rate"].dtype != float:
        print("\n\nError: MMR Vaccination Rate entries must be floats, "
              "not strings -- please fix this formatting.")
        return False
    else:
        return True


def check_vaccination_rates_nonzero(df):
    # We had some issues where non-responses were filled as "0% vaccination rate"
    #   -- this is why we need to check for this!

    if not (df["MMR Vaccination Rate"] > 1e-6).all():
        print("\n\nError: Some MMR Vaccination Rate entries are zero. Please check "
              "for non-responses mistakenly coded as empty, which is incorrect.")
        return False
    else:
        return True


def check_vaccination_rates_percentages(df):
    # Lauren's request: at most 2 decimal places

    try:
        vals_rounded_to_hundredths = np.round(df["MMR Vaccination Rate"] * 1000) / 1000

        if not (df["MMR Vaccination Rate"] == vals_rounded_to_hundredths).all():
            print("\n\nError: MMR Vaccination Rate entries must be expressed as "
                  "percents, not decimals. Each entry must be a float between "
                  "0 and 100, with at most two decimal places. Please fix this "
                  "formatting.")
            return False
        else:
            return True

    except:
        Exception("Exception: Could not apply rounding to MMR Vaccination Rate values.")
        return False


def check_vaccination_rate_CSV(filename: str):
    state_df = pd.read_csv(filename)

    if "Unnamed: 0" in state_df.columns:
        state_df.drop(columns="Unnamed: 0", inplace=True)

    columns_are_valid = check_columns_list(state_df)
    no_duplicates = check_no_duplicates(state_df)
    vaccination_rates_are_floats = check_vaccination_rates_floats(state_df)
    vaccination_rates_are_percentages = check_vaccination_rates_percentages(state_df)
    vaccination_rates_are_nonzero = check_vaccination_rates_nonzero(state_df)

    if (columns_are_valid &
            no_duplicates &
            vaccination_rates_are_floats &
            vaccination_rates_are_percentages &
            vaccination_rates_are_nonzero):
        print("This CSV appears valid and correctly formatted.")
    else:
        print(" !!! This CSV has validity or formatting issues -- please amend. !!! ")


def add_facility_address_to_school_district_or_name(df):

    # Identify duplicates
    duplicates = df[df.duplicated(subset=["School District or Name",
                                          "Age Group",
                                          "County"], keep=False)]

    # Append last two words from "Facility Address" to "School District or Name"
    df.loc[duplicates.index, "School District or Name"] = (
        df.loc[duplicates.index, "School District or Name"] + " " +
        df.loc[duplicates.index, "Facility Address"].apply(lambda x: " ".join(x.split()[-2:]))
    )

    return df


# %% Load data and run
if __name__ == "__main__":
    data_subfolder = 'state_data'
    filelist = [
        'NY_MMR_vax_rate.csv',
        'CT_MMR_vax_rate.csv',
        'MD_MMR_vax_rate.csv',
        'NM_MMR_vax_rate.csv',
        'WA_Rate_trial.csv'
        ]
    
    data_folder_path = os.sep.join([os.getcwd(), data_subfolder, ''])
    for csv_filename in filelist:
        if csv_filename in os.listdir(data_folder_path):
            print('\n\n#############################\n\n  Filename:', csv_filename)
            check_vaccination_rate_CSV(data_folder_path + csv_filename)
        else:
            print('File "' + csv_filename +'" not found.')


