pip install xgboost

# COMMAND ----------

pip install imblearn

# COMMAND ----------

# Import required libraries
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from xgboost import plot_importance
from matplotlib import pyplot
import sklearn
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import RandomOverSampler

# COMMAND ----------

# Read the train csv file into pandas dataframe
train_data = pandas_df = pd.read_csv("/dbfs/FileStore/tables/train_s3TEQDk/train_s3TEQDk.csv", header='infer')
# Read the test csv file into pandas dataframe
test_data = pandas_df = pd.read_csv("/dbfs/FileStore/tables/test_mSzZ8RL.csv", header='infer')

# Observation :::: 0a - We see the dataframe using below command
display(train_data.head())

# COMMAND ----------

#*******************************************Feature Engineering Starts**********************************************
# Check the data types of the columns
train_data.info()
# Observation :::: Since "Is_Lead" is a target column, I am not considering it to be treated as numerical column here.
#                 The reason being I will perform some numerical and categorical column manipulations. I intend to keep
#                 the target feature / label as intact 
# Again check the details of each column
train_data.info()
# Describe the dataframe to check the 5-number summary for each numerical columns. Note, "Is_Lead" is no more numerical now.
train_data.describe()

# COMMAND ----------

# Check unique values of each categorical column. If any variable has Nan as per its designed data type then it will be caught
# col_list = train_data.columns.tolist()
# Check if any data is Nan in the entire dataframe
# Observation :::: The below command returned TRUE indicating we have some NANs in our dataset
train_data.isnull().values.any()
# Print which columns has how many NANs I run the below command
sum_nan = train_data.isnull().sum()
print(sum_nan)
# Observation 3:::: I notice that we have 29325 records which has Credit_Product as NAN in train data
nan_rows = train_data[train_data['Credit_Product'].isnull()]
# Verify the length of rows with sum_nan so as to ensure that we dont have NAN due to any other columns
#display(len(nan_rows))
# Observation :::: 29325 is a significant number for me. Hence, I cannot drop these rows. 
# Also, I can not apply mode on the entire data set and impute all NANs with mode of entire dataset.

# Approach :::: I will first impute all NANs with literal "Unknown". So that I have a definite word to express them.
# Then later I will do clustering of my dataset based on foloowing features 
# Gender , Age, Region_Code, Occupation , Channel and Vinatge column.
# Upon clustering, I will get small groupded subset of datasets. If any "Unknown" will appear in them , it will be replaced with
# the mode of the cluster. This way I can club together the dataset at more granular level and get the most appropriate mode.

train_data.loc[(train_data["Credit_Product"] != train_data["Credit_Product"]),"Credit_Product"] = "Unknown"
# To generate clusters, I will not use any standard algorithm , rather join the required columns together
# Convert them as "category" and encode them. The resultant encoded number is my cluster number against the dataset row.
# Then I find the "mode" of each clustered group and replace the value of NAN in Credit_Product column with the mode of the cluster.

# Join Gender, Age, Region_Code, Occupation , Channel_Code and Vintage column
train_data["new_cluster"] = train_data["Gender"].map(str) +"_"+ train_data["Region_Code"].map(str) +"_"+ train_data["Occupation"].map(str) +"_"+ train_data["Channel_Code"].map(str) +"_"+ train_data["Vintage"].map(str) +"_"+ train_data["Age"].map(str)

# Encode the "new_cluster" column by treating it as "category" and converting it to numbers
train_data["encoded_new_cluster"] =  train_data["new_cluster"].astype("category").cat.codes

print(len(np.unique(train_data.encoded_new_cluster)))
print(train_data.shape)

# Observation :::: We get 22974 distinct clusters when we exclude AGE feature. With AGE feature it takes us to more granular level
# and generates 126566 clusters. 

# Approach ::: I decided to keep it as much granular as possible as that represents best fact about a cluster.
# After this encoded cluster using Gender , Age, Region_Code, Occupation , Channel and Vinatge columns , I will now take out
# mode of each of these clusters on Credit_Product column

# Calculate the mode for column Credit_Product for each cluster. Store it in a new dataframe
grouped_dataset = pd.DataFrame(train_data.groupby("encoded_new_cluster")["Credit_Product"].apply(pd.Series.mode)).reset_index()

df = grouped_dataset[["encoded_new_cluster","Credit_Product"]].groupby(['encoded_new_cluster']).agg(['count'])
df.columns = ["count"]
df.loc[df["count"] > 1]

# Observation :::: We notice that there are many clusters for which there are no specific modes , this is because every 
# Credit_Product value appears same number of times within that cluster
# Get all these clusters numbers
df = df.reset_index()
ambiguous_clusters = (df.loc[df["count"] > 1]["encoded_new_cluster"]).tolist()
# Observation :::: There are 3732 number of ambiguous clusters
print(len(ambiguous_clusters))

non_ambiguous_clusters = (df.loc[df["count"] == 1]["encoded_new_cluster"]).tolist()
# Observation :::: There are 19242 number of non ambiguous clusters
print(len(non_ambiguous_clusters))

cols = ["encoded_new_cluster","Credit_Product"]
grouped_dataset = grouped_dataset[cols]
display(grouped_dataset.head(2))

# Therefore, we can atleast , replace all the "Unknowns" for these 19242 clusters with their actual MODE values.
non_ambiguous_grouped_ds = grouped_dataset[grouped_dataset["encoded_new_cluster"].isin(non_ambiguous_clusters)]
print("Non-Ambiguous Length", len(non_ambiguous_grouped_ds))

# Also store the ambiguous clusters
ambiguous_grouped_ds = grouped_dataset[grouped_dataset["encoded_new_cluster"].isin(ambiguous_clusters)]
print("Ambiguous Length", len(ambiguous_grouped_ds))

# Observation :::: Totally they both should be equal in numbers to the length of grouped_dataset dataframe. Data integrity maintained :-)
print("Total Length", len(grouped_dataset))



# COMMAND ----------

# Loop through the training data and check if their Credit_Product value is "Unknown"
# If yes, then get the corrosponding row cluster number
# Get the matching corrosponding row cluster from non_ambiguous_grouped_ds
# If the matching row found, assign it the mode value.
# If not, it will continue to remain "Unknown"
for i in range(0,len(train_data)):
    credit_product = train_data['Credit_Product'].values[i]
    if(credit_product == "Unknown"):
        row_cluster = train_data['encoded_new_cluster'].values[i]
        mode_row = non_ambiguous_grouped_ds.loc[non_ambiguous_grouped_ds["encoded_new_cluster"] == row_cluster]
        if len(mode_row) > 0:
            mode = mode_row.Credit_Product.values[0]
            train_data['Credit_Product'].values[i] = mode

# COMMAND ----------

# Observation :::: Out of total 2,45,726 rows, and 29325 NAN values we merely now have 9140 rows 
# for which "Credit_Product" remains "Unknown". I will treat it as 3rd category for the Credit_Product feature.
# We have been greatly able to mitigate the issue which could have arised later in the code due to lost 29325 rows.
print(len(train_data.loc[train_data["Credit_Product"] == "Unknown"]))

# Rerun the code to affirm we have no more NANs in any of the columns. This is to also make sure that we had not 
# accidently generated new NANs


# Observation :::: All looks pretty good now. :-) 
sum_nan = train_data.isnull().sum()
print(sum_nan)

# We can now select only required columns of the dataframe
# Only keep the ones which we will use in our training.
col_list = ['Gender', 'Age', 'Region_Code', 'Occupation', 'Channel_Code','Vintage', 'Credit_Product', 'Avg_Account_Balance', 'Is_Active', 'Is_Lead']
train_data = train_data[col_list]

##### Creating some helper functions which will be later consumed.

# Generate coefficients of all numerical features of the dataframe. These coefficients will be used to further 
# quantize the values in the original dataframe
def generate_coefficients(df, pos_list):
    origin_list= []
    xgr = XGBClassifier(seed = 123, booster= 'gblinear', use_label_encoder=False)
    xgr.fit(df[pos_list], df['Is_Lead'])
    origin_list.append(pd.DataFrame(zip(df.columns,xgr.coef_)))
    origins_df = pd.concat(origin_list)
    origins_df.columns = ['cat_feat_name','value']
    origins_df["feature_name"] = origins_df['cat_feat_name']
    origins_df.index = origins_df['cat_feat_name']
    origins = origins_df.groupby('feature_name').transform('max')
    dr_fin = origins.drop_duplicates()
    dr_fin['norm_value'] = dr_fin['value']/dr_fin['value'].max()
    return dr_fin

# Define my ONE-HOT encoding method which converts all categorical features into numerical
def one_hot_encoding(df):
    # Categorical columns list
    cat_cols = ['Gender', 'Occupation', 'Region_Code', 'Channel_Code','Credit_Product', 'Is_Active']
    prefix_sep = '_'
    # Generate dummies using "get_dummies" method
    ohe_df = pd.get_dummies(data = df, columns = cat_cols, prefix = cat_cols, prefix_sep = prefix_sep)
    return ohe_df

# To normalize the numerical column, I have created below method
def normalize_feature(df,colname):
  min_of_col = df[colname].min()
  max_of_col = df[colname].max()
  norvalue = ((df[colname] - min_of_col)/ (max_of_col - min_of_col))
  return norvalue

# My initial approach to quantize my categorical columns was using below method, but somehow due to very high level of 
# granularity it did not perform well. Was taking more than 6 minutes to quantize all my distinct values under 
# each of my categorical columns
def quantization(df, coeff_df):
  # Treatment of categorical columns - Replacing the categorical values with their respective coefficients
  df["Gender"] = df["Gender"].apply(lambda x: coeff_df[coeff_df["cat_feat_name"] == "Gender_" + str(x)]['norm_value'].values[0])
  df["Region_Code"] = df["Region_Code"].apply(lambda x: coeff_df[coeff_df["cat_feat_name"] == "Region_Code_" + str(x)]['norm_value'].values[0])
  df["Occupation"] = df["Occupation"].apply(lambda x: coeff_df[coeff_df["cat_feat_name"] == "Occupation_" + str(x)]['norm_value'].values[0])
  df["Channel_Code"] = df["Channel_Code"].apply(lambda x: coeff_df[coeff_df["cat_feat_name"] == "Channel_Code_" + str(x)]['norm_value'].values[0])
  df["Credit_Product"] = df["Credit_Product"].apply(lambda x: coeff_df[coeff_df["cat_feat_name"] == "Credit_Product_" + str(x)]['norm_value'].values[0])
  df["Is_Active"] = df["Is_Active"].apply(lambda x: coeff_df[coeff_df["cat_feat_name"] == "Is_Active_" + str(x)]['norm_value'].values[0])
  return df


# Approach :::: I will generate dummies (on-hot encoding) for each of the categorical features.
train_data_numerical = one_hot_encoding(train_data)

# Pick up only the newly hot-encoded categorical columns
col_list= (train_data_numerical.columns[4:]).tolist()
# Append at the end of the list the label column which will be used to determine the coefficients using generate_coefficients method
col_list.append("Is_Lead")

train_data_numerical = train_data_numerical[col_list]
X_cols = (train_data_numerical.columns[:-1]).tolist()

# Set the label to be of type INT as for is required by the algorithm
train_data_numerical["Is_Lead"] = train_data_numerical["Is_Lead"].astype("int64")
# Generate coefficients and quantize the categorical columns
coeff_df = generate_coefficients(train_data_numerical, X_cols)
coeff_df = coeff_df[["cat_feat_name","norm_value"]]
# Reset the lable column to string.
train_data_numerical["Is_Lead"] = train_data_numerical["Is_Lead"].astype("str")

# Display the dataframe of all the coefficient features and their values
display(coeff_df)


# Get the coefficient for each of the feature above.

Gender_Female_coeff = coeff_df[coeff_df["cat_feat_name"] == "Gender_" + str("Female")]['norm_value'].values[0]
Gender_Male_coeff = coeff_df[coeff_df["cat_feat_name"] == "Gender_" + str("Male")]['norm_value'].values[0]

Occupation_Entrepreneur_coeff = coeff_df[coeff_df["cat_feat_name"] == "Occupation_" + str("Entrepreneur")]['norm_value'].values[0]
Occupation_Other_coeff = coeff_df[coeff_df["cat_feat_name"] == "Occupation_" + str("Other")]['norm_value'].values[0]
Occupation_Salaried_coeff = coeff_df[coeff_df["cat_feat_name"] == "Occupation_" + str("Salaried")]['norm_value'].values[0]
Occupation_Self_Employed_coeff = coeff_df[coeff_df["cat_feat_name"] == "Occupation_" + str("Self_Employed")]['norm_value'].values[0]

Channel_Code_X1_coeff = coeff_df[coeff_df["cat_feat_name"] == "Channel_Code_" + str("X1")]['norm_value'].values[0]
Channel_Code_X2_coeff = coeff_df[coeff_df["cat_feat_name"] == "Channel_Code_" + str("X2")]['norm_value'].values[0]
Channel_Code_X3_coeff = coeff_df[coeff_df["cat_feat_name"] == "Channel_Code_" + str("X3")]['norm_value'].values[0]
Channel_Code_X4_coeff = coeff_df[coeff_df["cat_feat_name"] == "Channel_Code_" + str("X4")]['norm_value'].values[0]

Credit_Product_No_coeff = coeff_df[coeff_df["cat_feat_name"] == "Credit_Product_" + str("No")]['norm_value'].values[0]
Credit_Product_Unknown_coeff = coeff_df[coeff_df["cat_feat_name"] == "Credit_Product_" + str("Unknown")]['norm_value'].values[0]
Credit_Product_Yes_coeff = coeff_df[coeff_df["cat_feat_name"] == "Credit_Product_" + str("Yes")]['norm_value'].values[0]

Is_Active_No_coeff = coeff_df[coeff_df["cat_feat_name"] == "Is_Active_" + str("No")]['norm_value'].values[0]
Is_Active_Yes_coeff = coeff_df[coeff_df["cat_feat_name"] == "Is_Active_" + str("Yes")]['norm_value'].values[0]

Region_Code_RG250_coeff = coeff_df[coeff_df["cat_feat_name"] == "Region_Code_" + str("RG250")]['norm_value'].values[0]
Region_Code_RG251_coeff = coeff_df[coeff_df["cat_feat_name"] == "Region_Code_" + str("RG251")]['norm_value'].values[0]
Region_Code_RG252_coeff = coeff_df[coeff_df["cat_feat_name"] == "Region_Code_" + str("RG252")]['norm_value'].values[0]
Region_Code_RG253_coeff = coeff_df[coeff_df["cat_feat_name"] == "Region_Code_" + str("RG253")]['norm_value'].values[0]
Region_Code_RG254_coeff = coeff_df[coeff_df["cat_feat_name"] == "Region_Code_" + str("RG254")]['norm_value'].values[0]
Region_Code_RG255_coeff = coeff_df[coeff_df["cat_feat_name"] == "Region_Code_" + str("RG255")]['norm_value'].values[0]
Region_Code_RG256_coeff = coeff_df[coeff_df["cat_feat_name"] == "Region_Code_" + str("RG256")]['norm_value'].values[0]
Region_Code_RG257_coeff = coeff_df[coeff_df["cat_feat_name"] == "Region_Code_" + str("RG257")]['norm_value'].values[0]
Region_Code_RG258_coeff = coeff_df[coeff_df["cat_feat_name"] == "Region_Code_" + str("RG258")]['norm_value'].values[0]
Region_Code_RG259_coeff = coeff_df[coeff_df["cat_feat_name"] == "Region_Code_" + str("RG259")]['norm_value'].values[0]
Region_Code_RG260_coeff = coeff_df[coeff_df["cat_feat_name"] == "Region_Code_" + str("RG260")]['norm_value'].values[0]
Region_Code_RG261_coeff = coeff_df[coeff_df["cat_feat_name"] == "Region_Code_" + str("RG261")]['norm_value'].values[0]
Region_Code_RG262_coeff = coeff_df[coeff_df["cat_feat_name"] == "Region_Code_" + str("RG262")]['norm_value'].values[0]
Region_Code_RG263_coeff = coeff_df[coeff_df["cat_feat_name"] == "Region_Code_" + str("RG263")]['norm_value'].values[0]
Region_Code_RG264_coeff = coeff_df[coeff_df["cat_feat_name"] == "Region_Code_" + str("RG264")]['norm_value'].values[0]
Region_Code_RG265_coeff = coeff_df[coeff_df["cat_feat_name"] == "Region_Code_" + str("RG265")]['norm_value'].values[0]
Region_Code_RG266_coeff = coeff_df[coeff_df["cat_feat_name"] == "Region_Code_" + str("RG266")]['norm_value'].values[0]
Region_Code_RG267_coeff = coeff_df[coeff_df["cat_feat_name"] == "Region_Code_" + str("RG267")]['norm_value'].values[0]
Region_Code_RG268_coeff = coeff_df[coeff_df["cat_feat_name"] == "Region_Code_" + str("RG268")]['norm_value'].values[0]
Region_Code_RG269_coeff = coeff_df[coeff_df["cat_feat_name"] == "Region_Code_" + str("RG269")]['norm_value'].values[0]
Region_Code_RG270_coeff = coeff_df[coeff_df["cat_feat_name"] == "Region_Code_" + str("RG270")]['norm_value'].values[0]
Region_Code_RG271_coeff = coeff_df[coeff_df["cat_feat_name"] == "Region_Code_" + str("RG271")]['norm_value'].values[0]
Region_Code_RG272_coeff = coeff_df[coeff_df["cat_feat_name"] == "Region_Code_" + str("RG272")]['norm_value'].values[0]
Region_Code_RG273_coeff = coeff_df[coeff_df["cat_feat_name"] == "Region_Code_" + str("RG273")]['norm_value'].values[0]
Region_Code_RG274_coeff = coeff_df[coeff_df["cat_feat_name"] == "Region_Code_" + str("RG274")]['norm_value'].values[0]
Region_Code_RG275_coeff = coeff_df[coeff_df["cat_feat_name"] == "Region_Code_" + str("RG275")]['norm_value'].values[0]
Region_Code_RG276_coeff = coeff_df[coeff_df["cat_feat_name"] == "Region_Code_" + str("RG276")]['norm_value'].values[0]
Region_Code_RG277_coeff = coeff_df[coeff_df["cat_feat_name"] == "Region_Code_" + str("RG277")]['norm_value'].values[0]
Region_Code_RG278_coeff = coeff_df[coeff_df["cat_feat_name"] == "Region_Code_" + str("RG278")]['norm_value'].values[0]
Region_Code_RG279_coeff = coeff_df[coeff_df["cat_feat_name"] == "Region_Code_" + str("RG279")]['norm_value'].values[0]
Region_Code_RG280_coeff = coeff_df[coeff_df["cat_feat_name"] == "Region_Code_" + str("RG280")]['norm_value'].values[0]
Region_Code_RG281_coeff = coeff_df[coeff_df["cat_feat_name"] == "Region_Code_" + str("RG281")]['norm_value'].values[0]
Region_Code_RG282_coeff = coeff_df[coeff_df["cat_feat_name"] == "Region_Code_" + str("RG282")]['norm_value'].values[0]
Region_Code_RG283_coeff = coeff_df[coeff_df["cat_feat_name"] == "Region_Code_" + str("RG283")]['norm_value'].values[0]
Region_Code_RG284_coeff = coeff_df[coeff_df["cat_feat_name"] == "Region_Code_" + str("RG284")]['norm_value'].values[0]



# Quantize all the categorical variables in the original train data
# Approach :::: Replace each of the distinct values of each categorical vairables with thier coefficients
def quantization(data):
    data["Gender"] = data["Gender"].apply(lambda g: Gender_Female_coeff if (g == "Female") else Gender_Male_coeff)
    data["Occupation"] = data["Occupation"].apply(lambda o: Occupation_Entrepreneur_coeff if (o == "Entrepreneur") else 
                                                            (Occupation_Salaried_coeff if (o == "Salaried") else 
                                                             (Occupation_Self_Employed_coeff if (o == "Self_Employed") else 
                                                           Occupation_Other_coeff)))
    data["Channel_Code"] = data["Channel_Code"].apply(lambda c: Channel_Code_X1_coeff if (c == "X1") else 
                                                            (Channel_Code_X2_coeff if (c == "X2") else 
                                                             (Channel_Code_X3_coeff if (c == "X3") else 
                                                              Channel_Code_X4_coeff)))
    data["Credit_Product"] = data["Credit_Product"].apply(lambda cp: Credit_Product_No_coeff if (cp == "No") else 
                                                            (Credit_Product_Yes_coeff if (cp == "Yes") else 
                                                             Credit_Product_Unknown_coeff))
    data["Is_Active"] = data["Is_Active"].apply(lambda a: Is_Active_No_coeff if (a == "No") else Is_Active_Yes_coeff)


    data["Region_Code"] = data["Region_Code"].apply(lambda rc: Region_Code_RG250_coeff if (rc == "RG250") else rc)
    data["Region_Code"] = data["Region_Code"].apply(lambda rc: Region_Code_RG251_coeff if (rc == "RG251") else rc)
    data["Region_Code"] = data["Region_Code"].apply(lambda rc: Region_Code_RG252_coeff if (rc == "RG252") else rc)
    data["Region_Code"] = data["Region_Code"].apply(lambda rc: Region_Code_RG253_coeff if (rc == "RG253") else rc)
    data["Region_Code"] = data["Region_Code"].apply(lambda rc: Region_Code_RG254_coeff if (rc == "RG254") else rc)
    data["Region_Code"] = data["Region_Code"].apply(lambda rc: Region_Code_RG255_coeff if (rc == "RG255") else rc)
    data["Region_Code"] = data["Region_Code"].apply(lambda rc: Region_Code_RG256_coeff if (rc == "RG256") else rc)
    data["Region_Code"] = data["Region_Code"].apply(lambda rc: Region_Code_RG257_coeff if (rc == "RG257") else rc)
    data["Region_Code"] = data["Region_Code"].apply(lambda rc: Region_Code_RG258_coeff if (rc == "RG258") else rc)
    data["Region_Code"] = data["Region_Code"].apply(lambda rc: Region_Code_RG259_coeff if (rc == "RG259") else rc)

    data["Region_Code"] = data["Region_Code"].apply(lambda rc: Region_Code_RG260_coeff if (rc == "RG260") else rc)
    data["Region_Code"] = data["Region_Code"].apply(lambda rc: Region_Code_RG261_coeff if (rc == "RG261") else rc)
    data["Region_Code"] = data["Region_Code"].apply(lambda rc: Region_Code_RG262_coeff if (rc == "RG262") else rc)
    data["Region_Code"] = data["Region_Code"].apply(lambda rc: Region_Code_RG263_coeff if (rc == "RG263") else rc)
    data["Region_Code"] = data["Region_Code"].apply(lambda rc: Region_Code_RG264_coeff if (rc == "RG264") else rc)
    data["Region_Code"] = data["Region_Code"].apply(lambda rc: Region_Code_RG265_coeff if (rc == "RG265") else rc)
    data["Region_Code"] = data["Region_Code"].apply(lambda rc: Region_Code_RG266_coeff if (rc == "RG266") else rc)
    data["Region_Code"] = data["Region_Code"].apply(lambda rc: Region_Code_RG267_coeff if (rc == "RG267") else rc)
    data["Region_Code"] = data["Region_Code"].apply(lambda rc: Region_Code_RG268_coeff if (rc == "RG268") else rc)
    data["Region_Code"] = data["Region_Code"].apply(lambda rc: Region_Code_RG269_coeff if (rc == "RG269") else rc)

    data["Region_Code"] = data["Region_Code"].apply(lambda rc: Region_Code_RG270_coeff if (rc == "RG270") else rc)
    data["Region_Code"] = data["Region_Code"].apply(lambda rc: Region_Code_RG271_coeff if (rc == "RG271") else rc)
    data["Region_Code"] = data["Region_Code"].apply(lambda rc: Region_Code_RG272_coeff if (rc == "RG272") else rc)
    data["Region_Code"] = data["Region_Code"].apply(lambda rc: Region_Code_RG273_coeff if (rc == "RG273") else rc)
    data["Region_Code"] = data["Region_Code"].apply(lambda rc: Region_Code_RG274_coeff if (rc == "RG274") else rc)
    data["Region_Code"] = data["Region_Code"].apply(lambda rc: Region_Code_RG275_coeff if (rc == "RG275") else rc)
    data["Region_Code"] = data["Region_Code"].apply(lambda rc: Region_Code_RG276_coeff if (rc == "RG276") else rc)
    data["Region_Code"] = data["Region_Code"].apply(lambda rc: Region_Code_RG277_coeff if (rc == "RG277") else rc)
    data["Region_Code"] = data["Region_Code"].apply(lambda rc: Region_Code_RG278_coeff if (rc == "RG278") else rc)
    data["Region_Code"] = data["Region_Code"].apply(lambda rc: Region_Code_RG279_coeff if (rc == "RG279") else rc)

    data["Region_Code"] = data["Region_Code"].apply(lambda rc: Region_Code_RG280_coeff if (rc == "RG280") else rc)
    data["Region_Code"] = data["Region_Code"].apply(lambda rc: Region_Code_RG281_coeff if (rc == "RG281") else rc)
    data["Region_Code"] = data["Region_Code"].apply(lambda rc: Region_Code_RG282_coeff if (rc == "RG282") else rc)
    data["Region_Code"] = data["Region_Code"].apply(lambda rc: Region_Code_RG283_coeff if (rc == "RG283") else rc)
    data["Region_Code"] = data["Region_Code"].apply(lambda rc: Region_Code_RG284_coeff if (rc == "RG284") else rc)
    return data
  
train_data = quantization(train_data)
display(train_data)

# As evident above columns' scale varies amongst themselves , hence let us normalize them , except Is_Lead since that is our target class/ variable.
# Normalize all numerical columns
col_list = ['Gender', 'Age', 'Region_Code', 'Occupation', 'Channel_Code', 'Vintage','Credit_Product', 'Avg_Account_Balance', 'Is_Active']
for col in col_list:
    train_data[col] = normalize_feature(train_data,col)
#*******************************************Feature Engineering Ends**********************************************


# Observation :::: All the numerical columns are now normalized.

# COMMAND ----------

X_data, y_data = train_data[col_list] ,train_data["Is_Lead"]
# Oversample the top selllers
ros = RandomOverSampler(random_state=10)
x_ros, y_ros = ros.fit_resample(X_data, y_data)
frames = [x_ros, y_ros]
balanced_train_data = pd.concat(frames, axis=1, join='inner')

# COMMAND ----------

#*******************************************Algorithm training Starts**********************************************

# Now let us start training our model on our balanced dataset
X, y = balanced_train_data[col_list] ,balanced_train_data["Is_Lead"]

# COMMAND ----------

# Find any highly correlated training features
corr_matrix = X.corr(method ='kendall').abs()
# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
# Find features with correlation greater than 0.95
to_drop = [column for column in upper.columns if any(upper[column] > 0.90)]
to_drop

# Observation :::: Looks like there are no highly correlated features

# COMMAND ----------

from sklearn import tree
tree_classifier = tree.DecisionTreeClassifier(max_depth = 22, random_state = 1).fit(X,y)
roc_auc_score(y, tree_classifier.predict(X), average=None)

# COMMAND ----------

from sklearn.ensemble import AdaBoostClassifier
model = AdaBoostClassifier(n_estimators = 250).fit(X, y)
roc_auc_score(y, model.predict(X), average=None)

# COMMAND ----------

X

# COMMAND ----------

model.feature_importances_
# Observation :::: Gender and Is_Active seems to be least important features

# COMMAND ----------

# Hence, carry only the important columns
imp_col_list = ['Age', 'Region_Code', 'Occupation', 'Vintage','Credit_Product', 'Avg_Account_Balance']

# COMMAND ----------

# Training only on important features
from sklearn.ensemble import AdaBoostClassifier
model = AdaBoostClassifier(n_estimators = 250).fit(X[imp_col_list], y)
roc_auc_score(y, model.predict(X[imp_col_list]), average=None)

# COMMAND ----------

model.feature_importances_

# COMMAND ----------

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(max_depth = 30, random_state = 0, n_estimators = 100).fit(X[imp_col_list], y)
roc_auc_score(y, rf.predict(X[imp_col_list]), average=None)

# COMMAND ----------

from sklearn.linear_model import RidgeClassifierCV
rdg = RidgeClassifierCV(alphas = 1.0, cv = None).fit(X[imp_col_list], y)
roc_auc_score(y, rdg.predict(X[imp_col_list]), average=None)

# COMMAND ----------

from sklearn.ensemble import GradientBoostingClassifier
model = GradientBoostingClassifier(n_estimators = 70, max_depth = 14, subsample = 0.8, learning_rate=0.3, random_state=10).fit(X, y)
roc_auc_score(y, model.predict(X), average=None)

# COMMAND ----------

test_data = pandas_df = pd.read_csv("/dbfs/FileStore/tables/test_mSzZ8RL.csv", header='infer')
test_data.describe()

# COMMAND ----------

test_data.loc[(test_data["Credit_Product"] != test_data["Credit_Product"]),"Credit_Product"] = "Unknown"

# COMMAND ----------

# Normalize all numerical columns
col_list = ['Gender', 'Age', 'Region_Code', 'Occupation', 'Channel_Code', 'Vintage','Credit_Product', 'Avg_Account_Balance', 'Is_Active']
for col in col_list:
    test_data[col] = normalize_feature(test_data,col)

# COMMAND ----------

labels = abc.predict(test_data[col_list])
test_data["Is_Lead"] = labels
