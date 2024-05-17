
# IMPORTS
import pandas as pd
import time
import matplotlib.pyplot as plt
from tqdm import tqdm





# aggregates competitors' rates and invs into two new columns and fills in the remaining missing values
def aggregate_comp(df):
    comp_columns = [c for c in df.columns if "comp" in c]
    inv_columns = [c for c in comp_columns if c.endswith("_inv")]
    rate_columns = [c for c in comp_columns if c.endswith("_rate")]
    # create the aggregated columns
    df["comp_rate_agg"] = 0
    df["comp_inv_agg"] = 0


    # mean of the rates
    df["comp_rate_agg"] = df[rate_columns].sum(axis=1)

    # where all values are missing set to 0
    df.loc[df[rate_columns].isna().all(axis=1), "comp_rate_agg"] = 0

    # mean of the invs
    df["comp_inv_agg"] = (df[inv_columns]-0.5).sum(axis=1) # subtract to center the values around 0

    # where all values are missing set to 0
    df.loc[df[inv_columns].isna().all(axis=1), "comp_inv_agg"] = 0

    assert df["comp_rate_agg"].isna().sum() == 0, "There are still missing values in the comp_rate_agg column"

    assert df["comp_inv_agg"].isna().sum() == 0, "There are still missing values in the comp_inv_agg column"


    df.drop(['comp1_rate','comp1_inv','comp1_rate_percent_diff','comp2_rate','comp2_inv','comp2_rate_percent_diff','comp3_rate','comp3_inv','comp3_rate_percent_diff','comp4_rate','comp4_inv','comp4_rate_percent_diff','comp5_rate','comp5_inv','comp5_rate_percent_diff','comp6_rate','comp6_inv','comp6_rate_percent_diff','comp7_rate','comp7_inv','comp7_rate_percent_diff','comp8_rate','comp8_inv','comp8_rate_percent_diff'],axis=1, inplace=True)

    return df





# ### Dealing with the visitor_hist_starrating and visitor_hist_adr_usd columns
# 
def fill_vistor_hist(df):

    df.drop(['visitor_hist_starrating'],axis=1, inplace=True)

    # count the number of missing values for visitor_hist_adr_usd
    hist_usd_missing= df["visitor_hist_adr_usd"].isna().sum()
    print(f"the number of missing values for visitor_hist_adr_usd is {hist_usd_missing}")

    mean = df["visitor_hist_adr_usd"].mean()

    df.groupby("visitor_location_country_id")["visitor_hist_adr_usd"].transform(lambda x: print(len(x), x.mean()))

    # fill the missing values of visitor_hist_adr_usd with the mean over visitor_location_country_id
    df["visitor_hist_adr_usd"] = df.groupby("visitor_location_country_id")["visitor_hist_adr_usd"].transform(lambda x: x.fillna(x.mean()))

    hist_usd_missing= df["visitor_hist_adr_usd"].isna().sum()
    print(f"the number of missing values for visitor_hist_adr_usd after first filling by country is {hist_usd_missing}")

    # fill the remaining missing values with the mean
    df["visitor_hist_adr_usd"] = df["visitor_hist_adr_usd"].fillna(mean)

    hist_usd_missing= df["visitor_hist_adr_usd"].isna().sum()
    print(f"the number of missing values for visitor_hist_adr_usd after second filling by mean is {hist_usd_missing}")

    return df

class VisitorHistFiller:
    def __init__(self):
        self.global_mean = None
        self.country_means = None

    def fit_transform(self, df):
        # Drop 'visitor_hist_starrating' column if exists
        if 'visitor_hist_starrating' in df.columns:
            df.drop(['visitor_hist_starrating'], axis=1, inplace=True)

         # count the number of missing values for visitor_hist_adr_usd
        hist_usd_missing= df["visitor_hist_adr_usd"].isna().sum()
        print(f"the number of missing values for visitor_hist_adr_usd is {hist_usd_missing}")


        # Compute the global mean
        self.global_mean = df["visitor_hist_adr_usd"].mean()

        # Compute the mean for each country
        self.country_means = df.groupby("visitor_location_country_id")["visitor_hist_adr_usd"].mean()

        # Fill the missing values by country mean
        df["visitor_hist_adr_usd"] = df.groupby("visitor_location_country_id")["visitor_hist_adr_usd"].transform(
            lambda x: x.fillna(x.mean())
        )

        hist_usd_missing= df["visitor_hist_adr_usd"].isna().sum()
        print(f"the number of missing values for visitor_hist_adr_usd after first filling by country is {hist_usd_missing}")

        # Fill any remaining missing values with the global mean
        df["visitor_hist_adr_usd"] = df["visitor_hist_adr_usd"].fillna(self.global_mean)

        hist_usd_missing= df["visitor_hist_adr_usd"].isna().sum()
        print(f"the number of missing values for visitor_hist_adr_usd after second filling by mean is {hist_usd_missing}")

        # No return needed, modifications are done in-place
        return df

    def transform(self, df_test):
        # Fill missing values by country mean, using the means from the training data

        print(f"the number of missing values for visitor_hist_adr_usd is {df_test['visitor_hist_adr_usd'].isna().sum()}")

        
        if self.country_means is not None:
            for country_id, mean in self.country_means.items():
                mask = df_test["visitor_location_country_id"] == country_id
                df_test.loc[mask, "visitor_hist_adr_usd"] = mean
        else:
            print("Warning: country_means is None. Did you forget to call fit_transform on the training data?")
        
        print(f"the number of missing values for visitor_hist_adr_usd after first filling by country is {df_test['visitor_hist_adr_usd'].isna().sum()}")

        # Fill any remaining missing values with the global mean
        if self.global_mean is not None:
            df_test["visitor_hist_adr_usd"] = df_test["visitor_hist_adr_usd"].fillna(self.global_mean)
        else:
            print("Warning: global_mean is None. Did you forget to call fit_transform on the training data?")

        final_missing = df_test["visitor_hist_adr_usd"].isna().sum()


        print(f"the number of missing values for visitor_hist_adr_usd after second filling by mean is {final_missing}")

        assert final_missing == 0, "There are still missing values in the visitor_hist_adr_usd column"
        
        # No return needed, modifications are done in-place
        return df_test