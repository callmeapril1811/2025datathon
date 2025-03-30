#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import itertools

df_syf_id = pd.read_csv('syf_id_20250325.csv')
df_transaction_fact = pd.read_csv('transaction_fact_20250325.csv')
df_wrld_stor_tran_fact = pd.read_csv('wrld_stor_tran_fact_20250325.csv')
df_fraud_claim_case = pd.read_csv("fraud_claim_case_20250325.csv")
df_fraud_claim_tran = pd.read_csv("fraud_claim_tran_20250325.csv")
account_dim = pd.read_csv("account_dim_20250325.csv")
df_statement_fact = pd.read_csv('statement_fact_20250325.csv')
rams_batch_cur = pd.read_csv('rams_batch_cur_20250325.csv')

# Set display options (optional)
pd.set_option('display.max_columns', None)


###############################
# 1. Cleaning syf_id_20250325.csv
###############################

# Copy dataframe (if you need to keep the original)
df_clean_syf_id = df_syf_id.copy()

# Check missing values and value counts (for reference)
print("SYF_ID missing values:\n", df_clean_syf_id.isna().sum())
print("SYF_ID confidence_level value counts:\n", df_clean_syf_id["confidence_level"].value_counts())

# Drop 'closed_date' column and convert 'open_date' to datetime
df_clean_syf_id = df_clean_syf_id.drop(columns=["closed_date"])
df_clean_syf_id["open_date"] = pd.to_datetime(df_clean_syf_id["open_date"], format="%Y-%m-%d")

# Write cleaned syf_id CSV
df_clean_syf_id.to_csv('syf_id_20250325_clean.csv', index=False)


###############################################
# 2. Cleaning transaction_fact_20250325.csv
###############################################

df_clean_transaction_fact = df_transaction_fact.copy()

# Check missing values and value counts (for reference)
print("Transaction Fact missing values:\n", df_clean_transaction_fact.isna().sum())
print("Transaction Fact frgn_curr_code value counts:\n", df_clean_transaction_fact["frgn_curr_code"].value_counts())

# Drop specified columns
cols_to_drop_tf = ["payment_type", "product_amt", "product_qty", "fcr_amount", "fcr_flag", "fcr_rate_of_exchange", "curr_markup_fee", "frgn_curr_code"]
df_clean_transaction_fact = df_clean_transaction_fact.drop(columns=cols_to_drop_tf)

# Convert date columns to datetime
df_clean_transaction_fact["transaction_date"] = pd.to_datetime(df_clean_transaction_fact["transaction_date"], format="%Y-%m-%d")
df_clean_transaction_fact["posting_date"] = pd.to_datetime(df_clean_transaction_fact["posting_date"], format="%Y-%m-%d")
df_clean_transaction_fact["adj_orgn_tran_dt"] = pd.to_datetime(df_clean_transaction_fact["adj_orgn_tran_dt"], format="%Y-%m-%d")

# Write cleaned transaction fact CSV
df_clean_transaction_fact.to_csv('transaction_fact_20250325_clean.csv', index=False)


###############################################
# 3. Cleaning wrld_stor_tran_fact_20250325.csv
###############################################

# Check dtypes and missing values for reference
print("World Store Tran Fact dtypes:\n", df_wrld_stor_tran_fact.dtypes)
print("World Store Tran Fact missing values:\n", df_wrld_stor_tran_fact.isna().sum())

# Drop specified columns
cols_to_drop_wrld = ["payment_type", "product_amt", "product_qty", "fcr_amount", "fcr_flag", "fcr_rate_of_exchange"]
df_clean_wrld_stor_tran_fact = df_wrld_stor_tran_fact.drop(columns=cols_to_drop_wrld)

# Check missing values after drop
print("After drop, missing values:\n", df_clean_wrld_stor_tran_fact.isna().sum())

# Replace foreign currency codes with country names using a dictionary
currency_dictionary = {
    '840': "United States dollar", '414': "Kuwaiti dinar", '978': "Euro", '826': "Pound sterling",
    '051': "Armenian dram", '124': "Canadian dollar", '484': "Mexican peso", '410': "South Korean won",
    '044': "Bahamian dollar", '144': "Sri Lankan rupee", '986': "Brazilian real", '682': "Saudi riyal",
    '356': "Indian rupee", '360': "Indonesian rupiah", '036': "Australian dollar", '752': "Swedish krona",
    '084': "Belize dollar", '608': "Philippine peso", '392': "Japanese yen", '936': "Ghanaian cedi",
    '352': "Icelandic króna", '320': "Guatemalan quetzal", '949': "Turkish lira", '504': "Moroccan dirham",
    '348': "Hungarian forint", '554': "New Zealand dollar", '756': "Swiss franc", '404': "Kenyan shilling",
    '188': "Costa Rican colon", '344': "Hong Kong dollar", '214': "Dominican peso", '032': "Argentine peso",
    '764': "Thai baht", '977': "Bosnia and Herzegovina convertible mark", '634': "Qatari riyal", '400': "Jordanian dinar",
    '388': "Jamaican dollar", '704': "Vietnamese đồng", '985': "Polish złoty", '376': "Israeli new shekel",
    '458': "Malaysian ringgit", '975': "Bulgarian lev", '604': "Peruvian sol", '901': "New Taiwan dollar",
    '578': "Norwegian krone", '524': "Nepalese rupee", '208': "Danish krone", '170': "Colombian peso",
    '784': "United Arab Emirates dirham", '586': "Pakistani rupee", '702': "Singapore dollar", '953': "CFP franc",
    '203': "Czech koruna", '050': "Bangladeshi taka", '156': "Renminbi", '710': "South African rand",
    '951': "East Caribbean dollar", '818': "Egyptian pound", '052': "Barbados dollar", '788': "Tunisian dinar",
    '340': "Honduran lempira", '152': "Chilean peso", '136': "Cayman Islands dollar"
}
df_clean_wrld_stor_tran_fact["frgn_curr_code"].replace(currency_dictionary, inplace=True)

# Replace erroneous values with NaN (if any)
df_clean_wrld_stor_tran_fact["frgn_curr_code"] = df_clean_wrld_stor_tran_fact["frgn_curr_code"].replace('\\"\\"', np.nan)

# Drop 'curr_markup_fee' column
df_clean_wrld_stor_tran_fact = df_clean_wrld_stor_tran_fact.drop(columns=["curr_markup_fee"])

# Convert date columns to datetime
df_clean_wrld_stor_tran_fact["transaction_date"] = pd.to_datetime(df_clean_wrld_stor_tran_fact["transaction_date"], format="%Y-%m-%d")
df_clean_wrld_stor_tran_fact["adj_orgn_tran_dt"] = pd.to_datetime(df_clean_wrld_stor_tran_fact["adj_orgn_tran_dt"], format="%Y-%m-%d")
df_clean_wrld_stor_tran_fact["posting_date"] = pd.to_datetime(df_clean_wrld_stor_tran_fact["posting_date"], format="%Y-%m-%d")

# Write cleaned wrld_stor_tran_fact CSV
df_clean_wrld_stor_tran_fact.to_csv('wrld_stor_tran_fact_20250325_clean.csv', index=False)


##################################################
# 4. Cleaning fraud_claim_case_20250325.csv
##################################################

# Print basic info (optional)
print("Fraud Claim Case shape:", df_fraud_claim_case.shape)
df_fraud_claim_case.info()

# Drop duplicates
df_fraud_claim_case.drop_duplicates(inplace=True)

# Write cleaned fraud_claim_case CSV
df_fraud_claim_case.to_csv('fraud_claim_case_20250325_clean.csv', index=False)


#############################################
# 5. Clean fraud_claim_tran_20250325.csv
#############################################

print("Fraud Claim Tran shape:", df_fraud_claim_tran.shape)
df_fraud_claim_tran.info()

# Drop duplicates
df_fraud_claim_tran.drop_duplicates(inplace=True)

# Save cleaned file
df_fraud_claim_tran.to_csv('fraud_claim_tran_20250325_clean.csv', index=False)


#############################################
# 6. Clean account_dim_20250325.csv
#############################################

print("Account Dim Head:")
print(account_dim.head())
print("Account Dim shape:", account_dim.shape)
print("Account Dim dtypes:\n", account_dim.dtypes)
print("Account Dim missing values:\n", account_dim.isna().sum())

# Clean account_dim based on given notes:
# Drop columns: 'overlimit_type_flag', 'special_finance_charge_ind', 'ext_status_reason_cd_desc', 'date_in_collection'
drop_columns = ['overlimit_type_flag', 'special_finance_charge_ind', 'ext_status_reason_cd_desc', 'date_in_collection']
account_dim = account_dim.drop(drop_columns, axis=1)

# Replace mysterious values with NaN
account_dim["ebill_ind"] = account_dim["ebill_ind"].replace('\\\\\\\\\\\\""', np.nan)
account_dim["card_activation_flag"] = account_dim["card_activation_flag"].replace('\\\\\\\\\\\\""', np.nan)
account_dim["payment_hist_1_12_mths"] = account_dim["payment_hist_1_12_mths"].replace('\\\\\\\\\\\\""', np.nan)
account_dim["payment_hist_13_24_mths"] = account_dim["payment_hist_13_24_mths"].replace('\\\\\\\\\\\\""', np.nan)

# Save cleaned account_dim file
account_dim.to_csv('account_dim_20250325_clean.csv', index=False)


#############################################
# 7. Clean statement_fact_20250325.csv
#############################################

df_clean_statement_fact = df_statement_fact.copy()

print("Statement Fact dtypes:\n", df_clean_statement_fact.dtypes)
print("Statement Fact missing values:\n", df_clean_statement_fact.isna().sum())

# For reference: value counts for return_check_cnt columns
print("return_check_cnt_ytd:\n", df_clean_statement_fact["return_check_cnt_ytd"].value_counts())
print("return_check_cnt_2yr:\n", df_clean_statement_fact["return_check_cnt_2yr"].value_counts())
print("return_check_cnt_total:\n", df_clean_statement_fact["return_check_cnt_total"].value_counts())
print("return_check_cnt_last_mth:\n", df_clean_statement_fact["return_check_cnt_last_mth"].value_counts())
print("Unique return_check_cnt_last_mth values:", list(df_clean_statement_fact["return_check_cnt_last_mth"].unique()))

# Drop 'return_check_cnt_last_mth' column
columns_to_drop_sf = ["return_check_cnt_last_mth"]
df_clean_statement_fact = df_clean_statement_fact.drop(columns=columns_to_drop_sf)

# Replace mysterious values in payment history with NaN
df_clean_statement_fact["payment_hist_1_12_mths"] = df_clean_statement_fact["payment_hist_1_12_mths"].replace('\\"\\"', np.nan)

# Convert billing_cycle_date to datetime
df_clean_statement_fact["billing_cycle_date"] = pd.to_datetime(df_clean_statement_fact["billing_cycle_date"], format="%Y-%m-%d")

# Save cleaned statement_fact file
df_clean_statement_fact.to_csv('statement_fact_20250325_clean.csv', index=False)

#############################################
# 7. Clean rams_batch_cur_20250325.csv
#############################################

rams_batch_cur = pd.read_csv('rams_batch_cur_20250325.csv')
rams_batch_cur.loc[rams_batch_cur["cu_cash_line_am"] == 999999999999999, "cu_cash_line_am"] = 0
rams_batch_cur.loc[rams_batch_cur["ca_cash_bal_pct_cash_line"] == 999, "ca_cash_bal_pct_cash_line"] = 0
rams_batch_cur.to_csv('rams_batch_cur_20250325_clean.csv', index=False)



print("All cleaning processes complete. Clean CSV files have been saved.")
