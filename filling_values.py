
# IMPORTS
import pandas as pd
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from scipy.stats import gamma

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV, Lasso
from sklearn.metrics import mean_squared_error

from sklearn.linear_model import LinearRegression






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



class VisitorHistFiller:
    """
    A class for filling missing values in the 'visitor_hist_adr_usd' column of and removing the 'visitor_hist_starrating' column from a DataFrame.

    Attributes:
        global_mean (float): The global mean value of 'visitor_hist_adr_usd'.
        country_means (pandas.Series): The mean values of 'visitor_hist_adr_usd' for each country.

    Methods:
        fit_transform(df): Fit the filler to the training data and transform the DataFrame.
        transform(df_test): Transform the test data using the fitted filler.
    """
    def __init__(self):
        self.global_mean = None
        self.country_means = None

    def fit_transform(self, df):
        # Drop 'visitor_hist_starrating' column if exists
        if 'visitor_hist_starrating' in df.columns:
            df = df.drop(['visitor_hist_starrating'], axis=1)
            print("dropped visitor_hist_starrating column")

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

        if 'visitor_hist_starrating' in df_test.columns:
            df_test = df_test.drop(['visitor_hist_starrating'], axis=1)
            print("dropped visitor_hist_starrating column")

        print(f"the number of missing values for visitor_hist_adr_usd is {df_test['visitor_hist_adr_usd'].isna().sum()}")

        
        if self.country_means is not None:
            for country_id, mean in self.country_means.items():
                mask = df_test["visitor_location_country_id"] == country_id
                df_test.loc[mask, "visitor_hist_adr_usd"] = df_test.loc[mask, "visitor_hist_adr_usd"].fillna(mean)
            
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
    

class PropReviewScoreFiller:
    """
    A class for filling missing values in the 'prop_review_score' column of a DataFrame.

    Attributes:
        prop_id_means (pandas.Series): The mean review score for each property ID.
        prop_starrating_means (pandas.Series): The mean review score for each property star rating.
        global_mean (float): The global mean review score.

    Methods:
        fit_transform(df): Fit the filler to the training data and transform the DataFrame.
        transform(df_test): Transform the test data using the fitted filler.
    """

    def __init__(self):
        # self.prop_id_means = None
        self.prop_starrating_means = None
        self.global_mean = None

    def fit_transform(self, df):
        # Print the initial count of missing values
        prop_review_missing = df["prop_review_score"].isna().sum()
        print(f"The number of missing values for prop_review_score is {prop_review_missing}")

        # # Compute the mean review score for each property ID
        # self.prop_id_means = df.groupby("prop_id")["prop_review_score"].mean()

        # # Fill missing review scores based on property ID
        # df["prop_review_score"] = df.groupby("prop_id")["prop_review_score"].transform(
        #     lambda x: x.fillna(x.mean())
        # )

        # # Update and print the count of missing values after filling by prop_id
        # prop_review_missing = df["prop_review_score"].isna().sum()
        # print(f"The number of missing values for prop_review_score after filling by prop_id is {prop_review_missing}")

        # Compute the mean review score for each property star rating
        self.prop_starrating_means = df.groupby("prop_starrating")["prop_review_score"].mean()

        # Fill remaining missing review scores based on property star rating
        df["prop_review_score"] = df.groupby("prop_starrating")["prop_review_score"].transform(
            lambda x: x.fillna(x.mean())
            
        )
        print(f"The number of missing values for prop_review_score after filling by prop_starrating is {df['prop_review_score'].isna().sum()}")

        # fill the remaining missing values with the global mean
        self.global_mean = df["prop_review_score"].mean()
        df["prop_review_score"] = df["prop_review_score"].fillna(self.global_mean)
    


        # Final update and print of missing values count
        prop_review_missing = df["prop_review_score"].isna().sum()
        print(f"The number of missing values for prop_review_score after filling by global mean is {prop_review_missing}")

        return df

    def transform(self, df_test):
        # Fill missing review scores based on property ID
        print(f"The number of missing values for prop_review_score is {df_test['prop_review_score'].isna().sum()}")
        # if self.prop_id_means is not None:
        #     for prop_id, mean in self.prop_id_means.items():
        #         mask = df_test["prop_id"] == prop_id
        #         df_test.loc[mask, "prop_review_score"].fillna(mean, inplace=True)
        #     print(f"The number of missing values for prop_review_score after filling by prop_id is {df_test['prop_review_score'].isna().sum()}")
        # else:
        #     print("Warning: prop_id_means is None. Did you forget to call fit_transform on the training data?")

        # Fill remaining missing review scores based on property star rating
        if self.prop_starrating_means is not None:
            for prop_starrating, mean in self.prop_starrating_means.items():
                mask = df_test["prop_starrating"] == prop_starrating
                df_test.loc[mask, "prop_review_score"] = df_test.loc[mask, "prop_review_score"].fillna(mean)
            print(f"The number of missing values for prop_review_score after filling by prop_starrating is {df_test['prop_review_score'].isna().sum()}")
        else:
            print("Warning: prop_starrating_means is None. Did you forget to call fit_transform on the training data?")
        
        # Fill the remaining missing values with the global mean
        if self.global_mean is not None:
            df_test["prop_review_score"] = df_test["prop_review_score"].fillna(self.global_mean)
        else:
            print("Warning: global_mean is None. Did you forget to call fit_transform on the training data?")

        final_missing = df_test["prop_review_score"].isna().sum()
        print(f"The number of missing values for prop_review_score after filling is {final_missing}")

        assert final_missing == 0, "There are still missing values in the prop_review_score column"

        return df_test
    

class DestinationDistanceFiller:
    """
    A class for filling missing values in the 'orig_destination_distance' column of a DataFrame, it iteratively fills missing values based on different groupings of the data.

    Attributes:
        global_mean (float): The global mean of 'orig_destination_distance'.
        means_by_country_destination (pandas.Series): The mean 'orig_destination_distance' for each pair of visitor location country ID and search destination ID.
        means_by_visitor_prop_country (pandas.Series): The mean 'orig_destination_distance' for each pair of visitor location country ID and property country ID.
        means_by_visitor_country (pandas.Series): The mean 'orig_destination_distance' for each visitor location country ID.
        means_by_prop_country (pandas.Series): The mean 'orig_destination_distance' for each property country ID.
        means_by_prop_id (pandas.Series): The mean 'orig_destination_distance' for each property ID.

    Methods:
        fit_transform(df): Fit the filler to the training data and transform the DataFrame.
        transform(df_test): Transform the test data using the fitted filler.
    """
    def __init__(self):
        self.global_mean = None
        self.means_by_country_destination = None
        self.means_by_visitor_prop_country = None
        self.means_by_visitor_country = None
        self.means_by_prop_country = None
        self.means_by_prop_id = None

    def fit_transform(self, df):
        # Calculate the global mean
        self.global_mean = df["orig_destination_distance"].mean()

        # Initial missing value count
        orig_missing = df["orig_destination_distance"].isna().sum()
        print(f"The number of missing values for orig_destination_distance is {orig_missing}")

        # Fill by visitor_location_country_id and srch_destination_id
        self.means_by_country_destination = df.groupby(["visitor_location_country_id", "srch_destination_id"])["orig_destination_distance"].mean()
        df["orig_destination_distance"] = df.groupby(["visitor_location_country_id", "srch_destination_id"])["orig_destination_distance"].transform(lambda x: x.fillna(x.mean()))
        print(f"The number of missing values after filling by visitor_location_country_id and srch_destination_id is {df['orig_destination_distance'].isna().sum()}")

        # Fill by visitor_location_country_id and prop_country_id
        self.means_by_visitor_prop_country = df.groupby(["visitor_location_country_id", "prop_country_id"])["orig_destination_distance"].mean()
        df["orig_destination_distance"] = df.groupby(["visitor_location_country_id", "prop_country_id"])["orig_destination_distance"].transform(lambda x: x.fillna(x.mean()))
        print(f"The number of missing values after filling by visitor_location_country_id and prop_country_id is {df['orig_destination_distance'].isna().sum()}")

        # Fill by visitor_location_country_id
        self.means_by_visitor_country = df.groupby(["visitor_location_country_id"])["orig_destination_distance"].mean()
        df["orig_destination_distance"] = df.groupby(["visitor_location_country_id"])["orig_destination_distance"].transform(lambda x: x.fillna(x.mean()))
        print(f"The number of missing values after filling by visitor_location_country_id is {df['orig_destination_distance'].isna().sum()}")

        # Fill by prop_country_id
        self.means_by_prop_country = df.groupby(["prop_country_id"])["orig_destination_distance"].mean()
        df["orig_destination_distance"] = df.groupby(["prop_country_id"])["orig_destination_distance"].transform(lambda x: x.fillna(x.mean()))
        print(f"The number of missing values after filling by prop_country_id is {df['orig_destination_distance'].isna().sum()}")

        # Fill by prop_id
        self.means_by_prop_id = df.groupby(["prop_id"])["orig_destination_distance"].mean()
        df["orig_destination_distance"] = df.groupby(["prop_id"])["orig_destination_distance"].transform(lambda x: x.fillna(x.mean()))
        print(f"The number of missing values after filling by prop_id is {df['orig_destination_distance'].isna().sum()}")

        # Fill remaining missing values with the global mean
        df["orig_destination_distance"] = df["orig_destination_distance"].fillna(self.global_mean)
        print(f"The number of missing values after filling by mean is {df['orig_destination_distance'].isna().sum()}")

        return df

    def transform(self, df_test):
        # Fill by visitor_location_country_id and srch_destination_id
        print(f"The number of missing values for orig_destination_distance is {df_test['orig_destination_distance'].isna().sum()}")
        if self.means_by_country_destination is not None:

            df_test["orig_destination_distance"] = df_test.groupby(["visitor_location_country_id", "srch_destination_id"])["orig_destination_distance"].transform(lambda x: x.fillna(value = self.means_by_country_destination.get(x.name)) if self.means_by_country_destination.get(x.name) else x)

            
            print(f"The number of missing values after filling by visitor_location_country_id and srch_destination_id is {df_test['orig_destination_distance'].isna().sum()}")

        else:
            print("Warning: means_by_country_destination is None. Did you forget to call fit_transform on the training data?")

        # Fill by visitor_location_country_id and prop_country_id
        if self.means_by_visitor_prop_country is not None:
            df_test["orig_destination_distance"] = df_test.groupby(["visitor_location_country_id", "prop_country_id"])["orig_destination_distance"].transform(lambda x: x.fillna(value = self.means_by_visitor_prop_country.get(x.name)) if self.means_by_visitor_prop_country.get(x.name) else x)
            print(f"The number of missing values after filling by visitor_location_country_id and prop_country_id is {df_test['orig_destination_distance'].isna().sum()}")
        else:
            print("Warning: means_by_visitor_prop_country is None. Did you forget to call fit_transform on the training data?")

        # Fill by visitor_location_country_id
        if self.means_by_visitor_country is not None:
            df_test["orig_destination_distance"] = df_test.groupby(["visitor_location_country_id"])["orig_destination_distance"].transform(lambda x: x.fillna(value = self.means_by_visitor_country.get(x.name)) if self.means_by_visitor_country.get(x.name) else x)
            print(f"The number of missing values after filling by visitor_location_country_id is {df_test['orig_destination_distance'].isna().sum()}")
        else:
            print("Warning: means_by_visitor_country is None. Did you forget to call fit_transform on the training data?")

        # Fill by prop_country_id
        if self.means_by_prop_country is not None:
            df_test["orig_destination_distance"] = df_test.groupby(["prop_country_id"])["orig_destination_distance"].transform(lambda x: x.fillna(value = self.means_by_prop_country.get(x.name)) if self.means_by_prop_country.get(x.name) else x)
            print(f"The number of missing values after filling by prop_country_id is {df_test['orig_destination_distance'].isna().sum()}")
        else:
            print("Warning: means_by_prop_country is None. Did you forget to call fit_transform on the training data?")

        # Fill by prop_id
        if self.means_by_prop_id is not None:
            df_test["orig_destination_distance"] = df_test.groupby(["prop_id"])["orig_destination_distance"].transform(lambda x: x.fillna(value = self.means_by_prop_id.get(x.name)) if self.means_by_prop_id.get(x.name) else x)
            print(f"The number of missing values after filling by prop_id is {df_test['orig_destination_distance'].isna().sum()}")
        else:
            print("Warning: means_by_prop_id is None. Did you forget to call fit_transform on the training data?")
        
        # Fill remaining missing values with the global mean
        df_test["orig_destination_distance"] = df_test["orig_destination_distance"].fillna(self.global_mean)

        final_missing = df_test["orig_destination_distance"].isna().sum()
        print(f"The number of missing values for orig_destination_distance after filling is {final_missing}")

        assert final_missing == 0, "There are still missing values in the orig_destination_distance column"

        return df_test


class QueryAffinityFiller:
    """
    A class for filling missing values in the 'srch_query_affinity_score' column of a DataFrame. The filler fits a gamma distribution to the positive values in the training data and samples from this distribution to fill the missing values.

    Attributes:
        alpha (float): The shape parameter of the fitted gamma distribution.
        beta (float): The scale parameter of the fitted gamma distribution.

    Methods:
        fit_transform(df): Fit the filler to the training data and transform the DataFrame.
        transform(df_test): Transform the test data using the fitted filler.
    """
    def __init__(self):
        self.alpha = None
        self.beta = None

    def fit_transform(self, df):
        # number of missing values
        total_missing = df["srch_query_affinity_score"].isna().sum()

        print(f"the number of missing values for srch_query_affinity_score is {total_missing}")
        positive_data = -df[['srch_query_affinity_score']].dropna()

        print("data converted to positive values")

        # Step 2: Fit a gamma distribution to the positive data
        self.alpha, loc, self.beta = gamma.fit(positive_data)  # We fix the location to 0 for simplicity

        print(f"alpha: {self.alpha}, loc: {loc}, beta: {self.beta}")

        # # Compute new mean (e.g., targeting the first percentile as new mean)
        target_mean = np.percentile(positive_data, 99)  # Assuming positive_data exists

        print(f"target mean: {target_mean}")
        self.alpha = target_mean / self.beta

        # convert to negative and to a pandas series
        sampled_data = -pd.Series(gamma.rvs(self.alpha, scale=self.beta, size=total_missing))

        print(f"the number of sampled values is {sampled_data.shape[0]}")
        print(f"the number of missing values is {df['srch_query_affinity_score'].isna().sum()}")
        print(f"the number of missing values in the sampled data is {sampled_data.isna().sum()}")

        missing_indices = df["srch_query_affinity_score"].isna()
        df.loc[missing_indices, "srch_query_affinity_score"] = sampled_data.values

        return df
    
    def transform(self, df_test):

        # number of missing values
        total_missing = df_test["srch_query_affinity_score"].isna().sum()

        print(f"the number of missing values for srch_query_affinity_score is {total_missing}")
        sampled_data = -pd.Series(gamma.rvs(self.alpha, scale=self.beta, size=total_missing))
        
        print(f"the number of missing values in the sampled data is {sampled_data.isna().sum()}")
        missing_indices = df_test["srch_query_affinity_score"].isna()

        df_test.loc[missing_indices, "srch_query_affinity_score"] = sampled_data.values

        missing_values = df_test["srch_query_affinity_score"].isna().sum()

        assert missing_values == 0, f"There are still {missing_values} missing values in the srch_query_affinity_score column"

        return df_test


class LocationScoreFiller:
    """
    A class for filling missing values in the 'prop_location_score2' column of a DataFrame. The filler uses a Lasso regression model to predict the missing values based on the other features in the data.

    Attributes:
        alpha (float): The regularization parameter of the Lasso model.
        selected_features (list): The features selected by the Lasso model.
        scaler (StandardScaler): The fitted StandardScaler object.
        linear_model (LinearRegression): The fitted LinearRegression model.

    Methods:
        fit_transform(df): Fit the filler to the training data and transform the DataFrame.
        transform(df_test): Transform the test data using the fitted filler.
    """
    def __init__(self):
        self.alpha = None
        self.selected_features = None
        self.scaler = None
        self.linear_model = None

    def fit_transform(self, df):

        no_missing_df = df.dropna(subset=["prop_location_score2"])

        # Select features and target variable
        X = no_missing_df.drop(['prop_location_score2','date_time', 'position', 'click_bool', 'booking_bool', 'random_bool', 'comp_rate_agg', 'comp_inv_agg'], axis=1)
        y = no_missing_df["prop_location_score2"]

        # normalize the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Initialize Lasso to select features
        model = Lasso(alpha=10**-3 * 3)
        model.fit(X_scaled, y)

        # Identify non-zero features
        selected_features = X.columns[model.coef_ != 0]

        # train the linear model on selected features
        X_selected = X[selected_features]
        X_selected_scaled = scaler.fit_transform(X_selected)

        # Initialize the model
        linear_model = LinearRegression()

        # Fit the model on the selected features
        linear_model.fit(X_selected_scaled, y)

        # Select rows with missing values
        missing_indices = df["prop_location_score2"].isna()
        X_missing = df.loc[missing_indices, selected_features]

        # Standardize the missing data
        X_missing_scaled = scaler.transform(X_missing)

        # Predict the missing values
        y_missing = linear_model.predict(X_missing_scaled)

        # Fill the missing values in the original DataFrame
        df.loc[missing_indices, "prop_location_score2"] = y_missing

        self.alpha = 10**-3 * 3
        self.selected_features = selected_features
        self.scaler = scaler
        self.linear_model = linear_model

        missing_values = df["prop_location_score2"].isna().sum()

        assert missing_values == 0, f"There are still {missing_values} missing values in the prop_location_score2 column"

        return df
    
    def transform(self, df_test):
            
        # Select rows with missing values
        missing_indices = df_test["prop_location_score2"].isna()
        X_missing = df_test.loc[missing_indices, self.selected_features]

        # Standardize the missing data
        X_missing_scaled = self.scaler.transform(X_missing)

        # Predict the missing values
        y_missing = self.linear_model.predict(X_missing_scaled)

        # Fill the missing values in the original DataFrame
        df_test.loc[missing_indices, "prop_location_score2"] = y_missing

        missing_values = df_test["prop_location_score2"].isna().sum()

        assert missing_values == 0, f"There are still {missing_values} missing values in the prop_location_score2 column"

        return df_test
          
    
# a class to run all the classes above in sequence with a fit_transform and transform method

class MissingValuesFiller:
    """
    A class for filling missing values in a DataFrame using a sequence of fillers.
    """
    def __init__(self):
        self.visitor_hist_filler = VisitorHistFiller()
        self.prop_review_score_filler = PropReviewScoreFiller()
        self.destination_distance_filler = DestinationDistanceFiller()
        self.query_affinity_filler = QueryAffinityFiller()
        self.location_score_filler = LocationScoreFiller()

    def fit_transform(self, df):
        print("Starting fit_transform process...")

        #remove gross_bookings_usd column if present
         
        if 'gross_bookings_usd' in df.columns:
            df = df.drop(['gross_bookings_usd'], axis=1)
            print("dropped gross_bookings_usd column")

        print("Aggregating comp columns..")
        df = aggregate_comp(df)


        # Process visitor history
        print("\nFilling visitor history ADR USD...")
        df = self.visitor_hist_filler.fit_transform(df)

        # Process property review scores
        print("\nFilling property review scores...")
        df = self.prop_review_score_filler.fit_transform(df)

        # Process destination distances
        print("\nFilling original destination distances...")
        df = self.destination_distance_filler.fit_transform(df)

        # Process search query affinity scores
        print("\nFilling search query affinity scores...")
        df = self.query_affinity_filler.fit_transform(df)

        # Process property location scores
        print("\nFilling property location scores...")
        df = self.location_score_filler.fit_transform(df)

        print("fit_transform process completed.")
        return df

    def transform(self, df_test):
        print("Starting transform process...")

        print("Aggregating comp columns..")
        df_test = aggregate_comp(df_test)

        # Process visitor history
        print("\nFilling visitor history ADR USD...")
        df_test = self.visitor_hist_filler.transform(df_test)

        # Process property review scores
        print("\nFilling property review scores...")
        df_test = self.prop_review_score_filler.transform(df_test)

        # Process destination distances
        print("\nFilling original destination distances...")
        df_test = self.destination_distance_filler.transform(df_test)

        # Process search query affinity scores
        print("\nFilling search query affinity scores...")
        df_test = self.query_affinity_filler.transform(df_test)

        # Process property location scores
        print("\nFilling property location scores...")
        df_test = self.location_score_filler.transform(df_test)

        print("transform process completed.")
        return df_test


