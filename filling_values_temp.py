
# IMPORTS
import pandas as pd
import time
import matplotlib.pyplot as plt
from tqdm import tqdm


# PARAMETERS

train_path = "data/basic_dataset.csv"
train_path = "data/training_set_VU_DM.csv"
# for testing: train_path = "data/small_dataset.csv"
save_output = True


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

    print(f"the number of nan values is {df["comp_inv_agg"].isna().sum()}")

    # where all values are missing set to 0
    df.loc[df[inv_columns].isna().all(axis=1), "comp_inv_agg"] = 0



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
        if self.country_means is not None:
            for country_id, mean in self.country_means.items():
                mask = df_test["visitor_location_country_id"] == country_id
                df_test.loc[mask, "visitor_hist_adr_usd"] = df_test.loc[mask, "visitor_hist_adr_usd"].fillna(mean)

        # Fill any remaining missing values with the global mean
        if self.global_mean is not None:
            df_test["visitor_hist_adr_usd"] = df_test["visitor_hist_adr_usd"].fillna(self.global_mean)

        # No return needed, modifications are done in-place
        return df_test

# %%

# get mean and stdv of prop_starrating for rows where prop_review_score is missing
mean = filled_df[filled_df.prop_review_score.isna()]["prop_starrating"].mean()
stdv = filled_df[filled_df.prop_review_score.isna()]["prop_starrating"].std()

print(f"mean: {mean}, stdv: {stdv}")

# %% [markdown]
# ### Dealing with prop_review_score

# %%

# fill prop_review_score by prop_id

prop_review_missing = filled_df["prop_review_score"].isna().sum()
print(f"the number of missing values for prop_review_score is {prop_review_missing}")

filled_df["prop_review_score"] = filled_df.groupby("prop_id")["prop_review_score"].transform(lambda x: x.fillna(x.mean()))

prop_review_missing = filled_df["prop_review_score"].isna().sum()
print(f"the number of missing values for prop_review_score after filling by prop_id is {prop_review_missing}")

# fill by prop_starrating
filled_df["prop_review_score"] = filled_df.groupby("prop_starrating")["prop_review_score"].transform(lambda x: x.fillna(x.mean()))

prop_review_missing = filled_df["prop_review_score"].isna().sum()
print(f"the number of missing values for prop_review_score after filling by prop_starrating is {prop_review_missing}")

# %%
# save the dataset that we have until now
filled_df.to_csv("data/half_filled_dataset.csv", index=False)

# %% [markdown]
# ## Checkpoint

# %%
half_filled_df = pd.read_csv("data/half_filled_dataset.csv")


# %% [markdown]
# ### Dealing with orig_destination_distance

# %%
# get mean of orig_destination_distance
mean = half_filled_df["orig_destination_distance"].mean()


# fill orig_destination_distance by visitor_location_country_id and srch_destination_id
orig_missing = half_filled_df["orig_destination_distance"].isna().sum()
print(f"the number of missing values for orig_destination_distance is {orig_missing}")

half_filled_df["orig_destination_distance"] = half_filled_df.groupby(["visitor_location_country_id", "srch_destination_id"])["orig_destination_distance"].transform(lambda x: x.fillna(x.mean()))

orig_missing = half_filled_df["orig_destination_distance"].isna().sum()
print(f"the number of missing values for orig_destination_distance after filling by visitor_location_country_id and srch_destination_id is {orig_missing}")

# fill the remaining missing values by visitor_location_country_id and prop_country_id

half_filled_df["orig_destination_distance"] = half_filled_df.groupby(["visitor_location_country_id", "prop_country_id"])["orig_destination_distance"].transform(lambda x: x.fillna(x.mean()))

orig_missing = half_filled_df["orig_destination_distance"].isna().sum()

print(f"the number of missing values for orig_destination_distance after filling by visitor_location_country_id and prop_country_id is {orig_missing}")

# fill the remaining missing values by visitor_location_country_id

half_filled_df["orig_destination_distance"] = half_filled_df.groupby(["visitor_location_country_id"])["orig_destination_distance"].transform(lambda x: x.fillna(x.mean()))

orig_missing = half_filled_df["orig_destination_distance"].isna().sum()

print(f"the number of missing values for orig_destination_distance after filling by visitor_location_country_id is {orig_missing}")

# fill the remaining missing values by prop_country_id

half_filled_df["orig_destination_distance"] = half_filled_df.groupby(["prop_country_id"])["orig_destination_distance"].transform(lambda x: x.fillna(x.mean()))

orig_missing = half_filled_df["orig_destination_distance"].isna().sum()

print(f"the number of missing values for orig_destination_distance after filling by prop_country_id is {orig_missing}")

# fill the remaining missing values by prop_id

half_filled_df["orig_destination_distance"] = half_filled_df.groupby(["prop_id"])["orig_destination_distance"].transform(lambda x: x.fillna(x.mean()))

orig_missing = half_filled_df["orig_destination_distance"].isna().sum()

print(f"the number of missing values for orig_destination_distance after filling by prop_id is {orig_missing}")

# fill the remaining by the mean
half_filled_df["orig_destination_distance"] = half_filled_df["orig_destination_distance"].fillna(mean)

orig_missing = half_filled_df["orig_destination_distance"].isna().sum()

print(f"the number of missing values for orig_destination_distance after filling by mean is {orig_missing}")



# %%
# save the dataset that we have until now
half_filled_df.to_csv("data/filled_dataset2.csv", index=False)

# %% [markdown]
# ### Dealing with srch_query_affinity_score

# %%
# fill srch_query_affinity_score by the min
half_filled_df = pd.read_csv("data/filled_dataset2.csv")

# calculate percentage of missing values
missing = half_filled_df["srch_query_affinity_score"].isna().sum()
percentage = missing / half_filled_df.shape[0] * 100

print(f"the number of missing values for srch_query_affinity_score is {missing}")
print(f"the percentage of missing values for srch_query_affinity_score is {percentage}")

# %%


# blot the distribution of srch_query_affinity_score
half_filled_df["srch_query_affinity_score"].hist(bins=100)

plt.show()

# %%
# remove outliers on the lower side
half_filled_df["srch_query_affinity_score"] = half_filled_df["srch_query_affinity_score"].apply(lambda x: x if x > -100 else None)

# blot the distribution of srch_query_affinity_score
half_filled_df["srch_query_affinity_score"].hist(bins=100)

plt.show()

# %%
from scipy.stats import gamma



# number of missing values
total_missing = half_filled_df["srch_query_affinity_score"].isna().sum()

print(f"the number of missing values for srch_query_affinity_score is {total_missing}")
positive_data = -half_filled_df[['srch_query_affinity_score']].dropna()

print("data converted to positive values")

# Step 2: Fit a gamma distribution to the positive data
alpha, loc, beta = gamma.fit(positive_data)  # We fix the location to 0 for simplicity



print(f"alpha: {alpha}, loc: {loc}, beta: {beta}")

# %%
# histogram of sample data from the distrinution the size should be the same as the number of non missing values
sample = gamma.rvs(alpha, loc=loc, scale=beta, size=positive_data.shape[0])

plt.hist(-sample, bins=100)
plt.show()

# %%
import numpy as np
from scipy.stats import gamma
import matplotlib.pyplot as plt

# Assume these are your original parameters
original_alpha = alpha
original_beta = beta

# Compute new mean (e.g., targeting the first percentile as new mean)
target_mean = np.percentile(positive_data, 99)  # Assuming positive_data exists

print(f"target mean: {target_mean}")
new_alpha = target_mean / beta

# Generate new adjusted data with the same variance but different mean
adjusted_data = gamma.rvs(new_alpha, scale=beta,loc=loc, size=positive_data.shape[0])

# Visual comparison
plt.figure(figsize=(12, 6))
plt.hist(positive_data, bins=30, alpha=0.5, label='Original Data', color='blue')
plt.hist(adjusted_data, bins=30, alpha=0.5, label='Adjusted Data', color='red')
plt.legend()
plt.title('Comparison of Original and Adjusted Data Distributions')
plt.show()


# %%
# Step 3: Sample new data points using the fitted parameters
sampled_data = gamma.rvs(new_alpha, scale=beta, size=total_missing)

# convert to negative and to a pandas series
sampled_data = -pd.Series(sampled_data)

print(f"the number of sampled values is {sampled_data.shape[0]}")
print(f"the number of missing values is {half_filled_df["srch_query_affinity_score"].isna().sum()}")
print(f"the number of missing values in the sampled data is {sampled_data.isna().sum()}")

missing_indices = half_filled_df["srch_query_affinity_score"].isna()
half_filled_df.loc[missing_indices, "srch_query_affinity_score"] = sampled_data.values


# %%
# plot the distribution of the filled srch_query_affinity_score
half_filled_df["srch_query_affinity_score"].hist(bins=100, density=True)    


# %%
# save the dataset that we have until now
half_filled_df.to_csv("data/filled_dataset3.csv", index=False)

# %% [markdown]
# ### Dealing with prop_location_score2

# %%
# check if location_score1 and location_score2 are correlated
half_filled_df = pd.read_csv("data/filled_dataset3.csv")

# calculate the correlation
correlation = half_filled_df["prop_location_score1"].corr(half_filled_df["prop_location_score2"])

print(f"the correlation between prop_location_score1 and prop_location_score2 is {correlation}")

# %%
# check if location_score2 can be predicted by location_score1 and other features

# drop the rows with missing values
loc_score_df = half_filled_df.dropna(subset=["prop_location_score2", "prop_location_score1"])

loc_score_df = loc_score_df.drop(['date_time', 'position', 'click_bool', 'gross_bookings_usd', 'booking_bool', 'random_bool', 'comp_rate_agg', 'comp_inv_agg'], axis=1)

# plot the correlation matrix of all features
import seaborn as sns

correlation_matrix = loc_score_df.corr()

plt.figure(figsize=(20, 20))

# format the correlation matrix with 2 decimal points
sns.heatmap(correlation_matrix, annot=True, fmt=".2f")

plt.show()





# %%
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV, Lasso
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Select features and target variable
X = loc_score_df.drop(["prop_location_score2"], axis=1)
y = loc_score_df["prop_location_score2"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize LassoCV
model = LassoCV(cv=5, random_state=42, n_alphas=50, n_jobs=4)

# Fit the LassoCV model
model.fit(X_train_scaled, y_train)

# Plot the coefficient path # set log scale for x-axis
plt.figure(figsize=(10, 6))
plt.semilogx()

# Display each feature's coefficient path

coefs = []
for a in model.alphas_:
    model_for_alpha = Lasso(alpha=a)
    model_for_alpha.fit(X_train_scaled, y_train)
    coefs.append(model_for_alpha.coef_)



plt.plot(model.alphas_, coefs)

# Highlight the optimal alpha
plt.axvline(x=model.alpha_, color='black', linestyle='--', label='Optimal Alpha')

alpha_used = 10**-3 * 3
plt.axvline(x=alpha_used, color='red', linestyle='--', label='Alpha used')

plt.xlabel('Alpha')
plt.ylabel('MSE')
plt.title('Lasso Paths')
plt.legend()
plt.axis('tight')
plt.semilogx()  # Change x-axis to log-scale for better visualization
plt.show()

# plot the mse for each alpha
plt.figure(figsize=(10, 6))
plt.semilogx() 
plt.plot(model.alphas_, model.mse_path_, linestyle='--')
plt.axvline(model.alpha_, linestyle='--', color='k', label='Optimal Alpha')
plt.axvline(alpha_used, color='red', linestyle='--', label='Alpha used')
plt.legend()
plt.xlabel('Alpha')
plt.ylabel('MSE')
plt.title('MSE for each Alpha')



# %%
# Evaluate the model on the test set
from sklearn.linear_model import LinearRegression

final_model = Lasso(alpha=alpha_used)
final_model.fit(X_train_scaled, y_train)
y_pred = final_model.predict(X_test_scaled)
test_mse = mean_squared_error(y_test, y_pred)
print(f"Test MSE: {test_mse}")

# Identify non-zero features and retrain model
selected_features = X.columns[final_model.coef_ != 0]
print("Selected features:", selected_features)

# Retrain Lasso model using only selected features
X_train_selected = X_train_scaled[:, final_model.coef_ != 0]
X_test_selected = X_test_scaled[:, final_model.coef_ != 0]


# fit a linear regression model

# Initialize the model
linear_model = LinearRegression()

# Fit the model on the selected features
linear_model.fit(X_train_selected, y_train)


# Evaluate the final model on the test set
final_y_pred = linear_model.predict(X_test_selected)
final_test_mse = mean_squared_error(y_test, final_y_pred)
print(f"Final Test MSE: {final_test_mse}")

# Visualize feature importance
plt.figure(figsize=(10, 6))
plt.bar( X.columns, np.abs(final_model.coef_))
plt.xlabel("Features")
plt.xticks(rotation=90)
plt.ylabel("Coefficient Magnitude")
plt.title("Feature Importance in Final Lasso Model")
plt.show()

# %%
# check the normality of the residuals
residuals = y_test - final_y_pred

plt.hist(residuals, bins=100)

# %%
# plot distribution of prop_location_score2
loc_score_df["prop_location_score2"].hist(bins=100)

# %%
from sklearn.linear_model import LinearRegression


# fit the model on the whole dataset and use it to predict the missing values
# Select features and target variable
X = loc_score_df.drop(["prop_location_score2"], axis=1)
X_selected = X[selected_features]

y = loc_score_df["prop_location_score2"]

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_selected)

# train the linear model above on the whole dataset
reg = LinearRegression().fit(X_scaled, y)


# Select rows with missing values
missing_indices = half_filled_df["prop_location_score2"].isna()
X_missing = half_filled_df.loc[missing_indices, selected_features]

# Standardize the missing data
X_missing_scaled = scaler.transform(X_missing)

# Predict the missing values
y_missing = reg.predict(X_missing_scaled)

# Fill the missing values in the original DataFrame
half_filled_df.loc[missing_indices, "prop_location_score2"] = y_missing




# %%
# plot the distribution of the filled prop_location_score2
half_filled_df["prop_location_score2"].hist(bins=100, density=True)

# %%
# print the number of missing values
missing = half_filled_df["prop_location_score2"].isna().sum()
print(f"the number of missing values for prop_location_score2 is {missing}")

# %%
# drop gross_bookings_usd
filled_df = half_filled_df.drop(['gross_bookings_usd'],axis=1)

# %%
# save the completed dataset
filled_df.to_csv("data/filled_dataset.csv", index=False)

# %% [markdown]
# # Test data

# %% [markdown]
# ### Setup

# %%
import pandas as pd
import dataset_utils as du

# %%
test_df = pd.read_csv("data/test_set_VU_DM.csv")

# %%
stats = du.missingness_and_uniqueness(test_df)
stats.to_csv("stats/test_missingness_stats.csv")

# %% [markdown]
# ### Similarity

# %%
# Things to plot to visualize the data:
# - comparison of distribution of each column, for booked and not booked rows
# - comparison of distribution of each column, for clicked and not clicked rows
# - distribution of sum rowwise, for clicked and not clicked rows
# - max rowwise, for clicked and not clicked rows
# - min rowwise, for clicked and not clicked rows
# - kurtosis rowwise, for clicked and not clicked rows
# - skewness rowwise, for clicked and not clicked rows
# - correlation matrix for all columns

# %% [markdown]
# # Outdated code

# %%
import numpy as np
from tqdm import tqdm

rate_columns = [c for c in df.columns if "rate" in c]
# iterate over the rows
cases_counts = np.zeros(5)
for i, row in tqdm(df.iterrows(), desc="rows"):

    #case 5
    if row[rate_columns].isna().all():
        df.loc[i, "comp_rate_agg"] = 0

        cases_counts[4] += 1

        continue
    
    # case 1
    
    case1 = all([row[c] == 0 or pd.isna(row[c]) for c in rate_columns])
    if case1:
        df.loc[i, "comp_rate_agg"] = 0

        cases_counts[0] += 1
        continue

    # case 2
    case2 = all([row[c] == 1 or pd.isna(row[c]) for c in rate_columns])
    if case2:

        df.loc[i, "comp_rate_agg"] = 1

        cases_counts[1] += 1
        continue

    # case 3

    case3 = all([row[c] == -1 or pd.isna(row[c]) for c in rate_columns])
    if case3:
        df.loc[i, "comp_rate_agg"] = -1

        cases_counts[2] += 1
        continue

    # case 4
    df.loc[i, "comp_rate_agg"] = row[rate_columns].mean()

    cases_counts[3] += 1


