## Homework 3 - Combined FSA Data Analysis
# Zane and Zach

##########################################
#                                        #
############### Libraries ################
#                                        #
########################################## 

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
import geopandas as gpd
from matplotlib.patches import Wedge

##########################################
#                                        #
############### Functions ################
#                                        #
########################################## 

# I asked AI to help code this after doing 5 of them "manually" - see below 
# for original code. AI chat log is included in submission files. 
def analyze_column(df, col_index):
    """Analyze a single column for missing values, unique counts, etc."""
    col = df.iloc[:, col_index]
    print(f"\n--- Column {col_index} Analysis ---")
    col_name = df.columns[col_index] if hasattr(df, 'columns') else f"Column {col_index}"
    print(f"\n{col_name}:")
    print(f"Missing values: {col.isna().sum()}")
    print(f"Unique values count: {len(col.unique())}")
    print(f"Non-null count: {col.count()}")
    print(f"Unique values: {col.unique()}")

# AI assisted code - see AI LOG 
def find_payment_at_sequence_end(fsa_data, payment_col='Disbursement Amount', dummy_col='Crisis Payment'):
    # Sort by payment amount in descending order (highest to lowest)
    df_sorted = fsa_data.sort_values(payment_col, ascending=False).reset_index(drop=True)
    
    # Find where col6 switches from 1 to 0
    for i in range(len(df_sorted) - 1):
        current_dummy = df_sorted.loc[i, dummy_col]
        next_dummy = df_sorted.loc[i + 1, dummy_col]
        
        # If we're in a sequence of 1's and the next value is 0
        if current_dummy == 1 and next_dummy == 0:
            return df_sorted.loc[i + 1, payment_col]  # Return the payment where it switches to 0
    
    # If no switch found, return None or handle edge case
    return None

def fix_fsa_code(code):
    """Fix FSA codes to proper format for mapping"""
    code_str = str(code)
    if len(code_str) == 1:
        return '1900' + code_str  # 1 → 19001
    elif len(code_str) == 2:
        return '190' + code_str   # 53 → 19053
    elif len(code_str) == 3:
        return '19' + code_str    # 153 → 19153
    else:
        return '19' + code_str    # Just in case

def assign_general_program(df):
    """Assign general program categories based on accounting program codes"""
    df['General Program'] = ""
    for i in range(len(df)):
        code = df.loc[i, 'Accounting Program Code']
        if code in [2775, 2832, 2833, 2835, 2875, 2881, 4042, 4056, 4058, 4060]:
            df.loc[i, 'General Program'] = "Disaster Support"
        elif code in [2837, 2838, 2840, 2862]:
            df.loc[i, 'General Program'] = "Commodity Support"
        elif code in [2867, 2868, 2877, 2878, 2879, 2880]:
            df.loc[i, 'General Program'] = "Trade War Relief"
        elif code in [2888, 2920, 3070, 3132, 3307, 3338, 3359, 3361]:
            df.loc[i, 'General Program'] = "Conservation"
        elif code in [4926, 4925, 4920, 4921]:
            df.loc[i, 'General Program'] = "COVID Relief"
        elif code in [6150, 6152]:
            df.loc[i, 'General Program'] = "Organic Support"
        elif code in [8025, 8053]:
            df.loc[i, 'General Program'] = "Dairy Support"
        else:
            df.loc[i, 'General Program'] = "Other"
    return df

def assign_payment_quarter(df):
    """Assign payment quarters based on payment date"""
    df['Payment Quarter'] = 0
    for i in range(len(df)):
        month = df.loc[i, 'Payment Date'].month
        if month in [1, 2, 3]:
            df.loc[i, 'Payment Quarter'] = 1
        elif month in [4, 5, 6]:
            df.loc[i, 'Payment Quarter'] = 2
        elif month in [7, 8, 9]:
            df.loc[i, 'Payment Quarter'] = 3
        elif month in [10, 11, 12]:
            df.loc[i, 'Payment Quarter'] = 4
    return df

##########################################
#                                        #
############## Data Import ###############
#                                        #
########################################## 

# Import data (Zach's PC)
# fsa_data = pd.read_excel("C:/Users/Zachary/OneDrive - University of Arkansas/MSECAN/Fall 2025/ISYS 51003 Data Analytics Fundamentals/HW3/iowa.xlsx")

# Import data (Zach's Laptop)
fsa_data = pd.read_excel("/Users/ztiptonuark.edu/Library/CloudStorage/OneDrive-UniversityofArkansas/MSECAN/Fall 2025/ISYS 51003 Data Analytics Fundamentals/HW3/iowa.xlsx")

# Test data subset for development
fsa_short = fsa_data.iloc[0:1000, :]

##########################################
#                                        #
######## Basic Data Exploration ##########
#                                        #
########################################## 

# Info table 
print(fsa_data.info())

# Look at first 10 values
print(fsa_data.head(10))

### Examine how many unique enteries per variable
# State FSA Code - 1 unique value (19)
print(fsa_data.iloc[:, 0].isna().sum())
print(len(fsa_data.iloc[:, 0].unique()))
print(fsa_data.iloc[:, 0].unique())

# State FSA Name - 1 unique value (Iowa)
print(fsa_data.iloc[:, 1].isna().sum())
print(len(fsa_data.iloc[:, 1].unique()))
print(fsa_data.iloc[:, 1].unique())

# County FSA Codes - 100 unique values
# Iowa is listed as having only 99 counties
# so this warrents further investigation
print(fsa_data.iloc[:, 2].isna().sum())
print(len(fsa_data.iloc[:, 2].unique()))
print(fsa_data.iloc[:, 2].unique())

# County FSA Names - 100 unique values
# One mystery solved, Pottawattamie is split 
# into West and East, so two counties
# Now to understand why that is 

# according to wikipedia, Pottawattamie 
# county has the highest corn production 
# in the US
print(fsa_data.iloc[:, 3].isna().sum())
print(len(fsa_data.iloc[:, 3].unique()))
print(fsa_data.iloc[:, 3].unique())

# Payee Name - 104450 unique records 
# with 722k rows that makes an average 
# of almost 7 programs per individual 
print(fsa_data.iloc[:, 4].isna().sum())
print(len(fsa_data.iloc[:, 4].unique()))
print(fsa_data.iloc[:, 4].unique())

# Address line - 106648 records with 12806 unique
# lots of missing data? or did they not fill in 
# repeat addresses? 
print(fsa_data.iloc[:, 5].isna().sum())
print(len(fsa_data.iloc[:, 5].unique()))
print(fsa_data.iloc[:, 5].count())
print(fsa_data.iloc[:, 5].unique())

# I think having an address line indicates a "larger" group
# I'm seeing LLC, Trust, Partnership, Estate, etc. with these 
# Not exactly sure how to test that everyone matches this
# so many records, but I think we'll run with the assumption 
has_address_info = fsa_data[fsa_data['Address Information Line'].notna()]
print(has_address_info.iloc[:, 4].unique())

# This is where I used the function defined above 
for i in range(16):
    analyze_column(fsa_data, i)

# Accounting Program year 579 Missing values 
missing_accounting_year = fsa_data[fsa_data['Accounting Program Year'].isna()]
# Dates are included in the name of the program 
# Interesting 
print(missing_accounting_year.iloc[:, 14].unique())

# Assigning an Accounting Program year by the name of the program
# got to work at small scale 
for i in missing_accounting_year.index:
    if missing_accounting_year.loc[i, 'Accounting Program Description'] == 'ECP/HMC COST SHARE FY19':
        missing_accounting_year.loc[i, 'Accounting Program Year'] = 2019
    elif missing_accounting_year.loc[i, 'Accounting Program Description'] == 'EMERGENCY CONSERVATION PROGRAM FY17':
        missing_accounting_year.loc[i, 'Accounting Program Year'] = 2017
    elif missing_accounting_year.loc[i, 'Accounting Program Description'] == 'ECP COST SHARE FY 2018':
        missing_accounting_year.loc[i, 'Accounting Program Year'] = 2018
    else:
        continue

# Now to try on large data set     
for i in fsa_data.index:
    if fsa_data.loc[i, 'Accounting Program Description'] == 'ECP/HMC COST SHARE FY19':
        fsa_data.loc[i, 'Accounting Program Year'] = 2019
    elif fsa_data.loc[i, 'Accounting Program Description'] == 'EMERGENCY CONSERVATION PROGRAM FY17':
        fsa_data.loc[i, 'Accounting Program Year'] = 2017
    elif fsa_data.loc[i, 'Accounting Program Description'] == 'ECP COST SHARE FY 2018':
        fsa_data.loc[i, 'Accounting Program Year'] = 2018
    else:
        continue

# Check for missing data 
# 5 entries all under "ECPCOF"
# only total of 55278 in disbursement amount - can drop if needed to 
print(fsa_data[fsa_data['Accounting Program Year'].isna()])
missing_accounting_year2 = fsa_data[fsa_data['Accounting Program Year'].isna()]

##########################################
#                                        #
########## Data Preprocessing ############
#                                        #
########################################## 

# Dropping State FSA Code (all 19)
# Dropping State FSA Name (all Iowa)
# This will help lower the RAM useage on such 
# a big dataset 
fsa_data = fsa_data.drop(['State FSA Code', 'State FSA Name'], axis = 1)

# Assign general program categories
fsa_data = assign_general_program(fsa_data)

# Create dummy variable for Crisis (COVID or TradeWar)
fsa_data['Crisis Payment'] = ((fsa_data['General Program'] == 'COVID Relief') | (fsa_data['General Program'] == "Trade War Relief")).astype(int)

print(fsa_data['Crisis Payment'].value_counts())

# Assign payment quarters
fsa_data = assign_payment_quarter(fsa_data)

##########################################
#                                        #
######## Descriptive Statistics ##########
#                                        #
########################################## 

### Descriptive Stats for whole data set 
pd.set_option('display.float_format', '{:.2f}'.format)
print(fsa_data['Disbursement Amount'].describe())
print(fsa_data['Disbursement Amount'].sum())

# Query by quartiles 
disbursement_Q1 = fsa_data[fsa_data['Disbursement Amount'] < 408]

plt.scatter(disbursement_Q1['Disbursement Amount'], disbursement_Q1['Accounting Program Code'])
plt.xlabel('$')
plt.ylabel('Program Code')
plt.title('Scatter Plot Title')
plt.show()

disbursement_Q2 = fsa_data[(fsa_data['Disbursement Amount'] >= 408) & (fsa_data['Disbursement Amount'] < 1165)]

plt.scatter(disbursement_Q2['Disbursement Amount'], disbursement_Q2['Accounting Program Code'])
plt.xlabel('$')
plt.ylabel('Program Code')
plt.title('Scatter Plot Title')
plt.show()

disbursement_Q3 = fsa_data[(fsa_data['Disbursement Amount'] >= 1165) & (fsa_data['Disbursement Amount'] < 3409)]

plt.scatter(disbursement_Q3['Disbursement Amount'], disbursement_Q3['Accounting Program Code'])
plt.xlabel('$ in dollars')
plt.ylabel('Program Code')
plt.title('Payment Distribution Plot for First 3 Quartiles')
plt.show()

disbursement_Q4 = fsa_data[(fsa_data['Disbursement Amount'] >= 3409)]
plt.clf()
plt.scatter(disbursement_Q4['Disbursement Amount'], disbursement_Q4['Accounting Program Code'])
plt.xlabel('$ in millions')
plt.ylabel('Program Code')
plt.title('Q4 Payment Distribution')
plt.show()

## some count stuff I did to check out pieces of the data 
count_under_100 = (fsa_data['Disbursement Amount'] < 100).sum()
print(f"Count of disbursements < $100: {count_under_100}")

count_under_10 = (fsa_data['Disbursement Amount'] < 10).sum()
print(f"Count of disbursements < $10: {count_under_10}")

count_under_1 = (fsa_data['Disbursement Amount'] < 1).sum()
print(f"Count of disbursements < $1: {count_under_1}")

count_over_5000 = (fsa_data['Disbursement Amount'] > 5000).sum()
print(f"Count of disbursements > $5000: {count_over_5000}")

count_over_10000 = (fsa_data['Disbursement Amount'] > 10000).sum()
print(f"Count of disbursements > $10000: {count_over_10000}")

count_over_100000 = (fsa_data['Disbursement Amount'] > 100000).sum()
print(f"Count of disbursements > $10000: {count_over_100000}")

count_over_1000000 = (fsa_data['Disbursement Amount'] > 1000000).sum()
print(f"Count of disbursements > $1,000,000: {count_over_1000000}")

disbursement_1mil = fsa_data[fsa_data['Disbursement Amount'] > 1000000]
print(disbursement_1mil)

## Query by "theme"
# This was AI assisted code - chat in AI LOG.docx
unique_pairs = fsa_data.drop_duplicates(subset=['Accounting Program Code', 'Accounting Program Description'])
print(unique_pairs[['Accounting Program Code', 'Accounting Program Description']])

# unique_pairs.to_csv('code_description_pairs.csv', index = False)

# I found this "in" method online and that I need to use `` with column names with 
# special characters and spaces 
disaster_sup = fsa_data.query("`Accounting Program Code` in [2775, 2832, 2833, 2835, 2875, 2881, 4042, 4056, 4058, 4060]")
commodity_sup = fsa_data.query("`Accounting Program Code` in [2837, 2838, 2840, 2862]")
trade_war_relief = fsa_data.query("`Accounting Program Code` in [2867, 2868, 2877, 2878, 2879, 2880]")
conservation = fsa_data.query("`Accounting Program Code` in [2888, 2920, 3070, 3132, 3307, 3338, 3359, 3361]")
COVID_relief = fsa_data.query("`Accounting Program Code` in [4926, 4925, 4920, 4921]")
organic_sup = fsa_data.query("`Accounting Program Code` in [6150, 6152]")
dairy_sup = fsa_data.query("`Accounting Program Code` in [8025, 8053, 8053]")

# new plots with general program instead of accounting code 
disbursement_Q1 = fsa_data[fsa_data['Disbursement Amount'] < 408]

plt.scatter(disbursement_Q1['Disbursement Amount'], disbursement_Q1['General Program'])
plt.xlabel('$ in dollars')
plt.ylabel('Genearl Program')
plt.title('Payment Distribution Plot for Quartiles 1-3')
plt.show()

disbursement_Q2 = fsa_data[(fsa_data['Disbursement Amount'] >= 408) & (fsa_data['Disbursement Amount'] < 1165)]

plt.scatter(disbursement_Q2['Disbursement Amount'], disbursement_Q2['General Program'])
plt.xlabel('$')
plt.ylabel('Program Code')
plt.title('Scatter Plot Title')
plt.show()

disbursement_Q3 = fsa_data[(fsa_data['Disbursement Amount'] >= 1165) & (fsa_data['Disbursement Amount'] < 3409)]

plt.scatter(disbursement_Q3['Disbursement Amount'], disbursement_Q3['General Program'])
plt.xlabel('$ in dollars')
plt.ylabel('Program Code')
plt.title('Payment Distribution Plot for First 3 Quartiles')
plt.show()

disbursement_Q4 = fsa_data[(fsa_data['Disbursement Amount'] >= 3409)]
plt.clf()
plt.scatter(disbursement_Q4['Disbursement Amount'], disbursement_Q4['General Program'])
plt.xlabel('$ in millions')
plt.ylabel('Program Code')
plt.title('Q4 Payment Distribution')
plt.show()

##########################################
#                                        #
########## Difference in Means ###########
#                                        #
########################################## 

# Basic t-test for differences in means
Crisis_payments = fsa_data[fsa_data['Crisis Payment'] == 1]['Disbursement Amount']
Normal_payments = fsa_data[fsa_data['Crisis Payment'] == 0]['Disbursement Amount']

# Log transformation - very right skewed data 
crisis_log = np.log(Crisis_payments)
normal_log = np.log(Normal_payments)

t_stat, p_value = stats.ttest_ind(crisis_log, normal_log, equal_var = False)
print(f"T stat {t_stat:.4f}, P value {p_value:.4f}")

# Run function to find top sums of Crisis Payments
result = find_payment_at_sequence_end(fsa_data, 'Disbursement Amount', 'Crisis Payment')
print(result)

top_end_disbursement = fsa_data[fsa_data['Disbursement Amount'] > 783909]
print(top_end_disbursement)
top_end_disbursement.to_csv('ted.csv')

print(top_end_disbursement['Disbursement Amount'].sum())
print(fsa_data['Disbursement Amount'].sum())
print(Crisis_payments.sum())
print(Normal_payments.sum())

fsa_data[fsa_data['General Program'] == 'COVID Relief']['Disbursement Amount'].sum()
fsa_data[fsa_data['General Program'] == 'Trade War Relief']['Disbursement Amount'].sum()

# Looking at non-crisis programs 
crisis_data = fsa_data[fsa_data['Crisis Payment'] == 1]
sans_crisis = fsa_data[fsa_data['Crisis Payment'] != 1]

crisis_data['Disbursement Amount'].describe()
sans_crisis['Disbursement Amount'].describe()

# pulling data from the companies that received individual payments over $1 million
# was it crisis only? 
titan_swine = (fsa_data[fsa_data['Formatted Payee Name'] == "TITAN SWINE"])
# titan_swine.to_csv('titanswine.csv')

d2k = (fsa_data[fsa_data['Formatted Payee Name'] == "D2K"])
# d2k.to_csv('D2K.csv')

hdp = (fsa_data[fsa_data['Formatted Payee Name'] == "H DIAMOND PARTNERS"])
# hdp.to_csv('hdp.csv')

nep = (fsa_data[fsa_data['Formatted Payee Name'] == "NEW ERA PARTNERSHIP"])
# nep.to_csv('nep.csv')

dsf = (fsa_data[fsa_data['Formatted Payee Name'] == "DOUG STUDER FARMS"])
# dsf.to_csv('dsf.csv')

crisis_table = (fsa_data[fsa_data['Crisis Payment'] == 1])
non_crisis_table = (fsa_data[fsa_data['Crisis Payment'] == 0])

# Find matching rows and then exclude them - AI CODE
# Add a source column to track which table each row came from
only_in_crisis = crisis_table[~crisis_table['Formatted Payee Name'].isin(non_crisis_table['Formatted Payee Name'])].copy()
only_in_crisis['Source'] = 'Crisis Only'

only_in_non_crisis = non_crisis_table[~non_crisis_table['Formatted Payee Name'].isin(crisis_table['Formatted Payee Name'])].copy()
only_in_non_crisis['Source'] = 'Non-Crisis Only'

# Combine them
unique_values = pd.concat([only_in_crisis, only_in_non_crisis], ignore_index=True)
unique_values.head()
print(unique_values['Formatted Payee Name'].nunique())
print(fsa_data['Formatted Payee Name'].nunique())

unique_values_no_dups = unique_values.drop_duplicates(subset='Formatted Payee Name')

print(unique_values_no_dups['Source'].value_counts())
print(unique_values_no_dups.groupby('Source')['Disbursement Amount'].sum())

unique_values_no_dups.to_csv('UVND.csv')

##########################################
#                                        #
############ Violin Plots ################
#                                        #
########################################## 

# Looking at non-crisis programs 
sans_crisis.describe()
sans_crisis.groupby('Payment Quarter').size()
plt.yscale('log')
ax = sns.violinplot(data=sans_crisis, x="Payment Quarter", y="Disbursement Amount", order = [1, 2, 3, 4])
# Calculate n for each quarter AI CODE
n_values = sans_crisis.groupby("Payment Quarter").size()
# Create new labels with n values AI CODE
quarter_order = [1, 2, 3, 4]  # Match the order parameter
new_labels = [f'{quarter}\n(n={n_values[quarter]})' for quarter in quarter_order if quarter in n_values]
ax.set_xticklabels(new_labels)
ax.tick_params(axis='both', which='major', labelsize=22)
plt.show()
# Add title and axis labels
plt.title("FSA Non-Crisis Support Disbursement Distribution by Quarter", fontsize=28, pad=20)
plt.xlabel("Payment Quarter", fontsize=22)
plt.ylabel("Disbursement Amount ($)", fontsize=22)
# Optional: Format y-axis tick labels AI CODE
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
plt.tight_layout()
plt.show()

# Crisis Program Plot
crisis_data.groupby('Payment Quarter').size()

plt.yscale('log')
ax = sns.violinplot(data=crisis_data, x="Payment Quarter", y="Disbursement Amount")

# Add title and axis labels
plt.title("FSA Crisis Support Disbursement Distribution by Quarter", fontsize=28, pad=20)
plt.xlabel("Payment Quarter", fontsize=22)
plt.ylabel("Disbursement Amount ($)", fontsize=22)

# Add n values to x-axis labels
n_values2 = crisis_data.groupby("Payment Quarter").size()
new_labels2 = [f'{quarter}\n(n={n_values2[quarter]})' for quarter in quarter_order if quarter in n_values2]
ax.set_xticklabels(new_labels2)
ax.tick_params(axis='both', which='major', labelsize=22)

# Format y-axis tick labels
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

plt.tight_layout()
plt.show()

# Counts
pd.set_option('display.float_format', '{:.2f}'.format)
sans_crisis.groupby('Payment Quarter')['Disbursement Amount'].agg(['count', 'sum', 'mean']).sort_index()
crisis_data.groupby('Payment Quarter')['Disbursement Amount'].agg(['count', 'sum', 'mean']).sort_index()

##########################################
#                                        #
############ ANOVA by Quarter ############
#                                        #
########################################## 

# ANOVA to test multiple groups 
# We have 3 groups for part time, full time, and per event
# I want to compare pay between them

sans_crisis_Q1 = sans_crisis[sans_crisis['Payment Quarter'] == 1]['Disbursement Amount']
sans_crisis_Q2 = sans_crisis[sans_crisis['Payment Quarter'] == 2]['Disbursement Amount']
sans_crisis_Q3 = sans_crisis[sans_crisis['Payment Quarter'] == 3]['Disbursement Amount']
sans_crisis_Q4 = sans_crisis[sans_crisis['Payment Quarter'] == 4]['Disbursement Amount']

# Use scipy to run a one-way ANOVA
# Is there a difference in pay between the three groups? 
# ANOVA will give us an f_stat and a p_value 

f_stat, p_value = stats.f_oneway(sans_crisis_Q1, sans_crisis_Q2, sans_crisis_Q3, sans_crisis_Q4)
print(f"P value {p_value:.4f} F Stat {f_stat:.2f}")

# ANOVA for crisis_data by quarter
crisis_data_Q1 = crisis_data[crisis_data['Payment Quarter'] == 1]['Disbursement Amount']
crisis_data_Q2 = crisis_data[crisis_data['Payment Quarter'] == 2]['Disbursement Amount']
crisis_data_Q3 = crisis_data[crisis_data['Payment Quarter'] == 3]['Disbursement Amount']
crisis_data_Q4 = crisis_data[crisis_data['Payment Quarter'] == 4]['Disbursement Amount']

# Use scipy to run a one-way ANOVA
# Is there a difference in pay between the three groups? 
# ANOVA will give us an f_stat and a p_value 

f_stat, p_value = stats.f_oneway(crisis_data_Q1, crisis_data_Q2, crisis_data_Q3, crisis_data_Q4)
print(f"P value {p_value:.4f} F Stat {f_stat:.2f}")

##########################################
#                                        #
############ Geographic Mapping ##########
#                                        #
########################################## 

# Create map data subset for better performance
map_data = fsa_data.drop(fsa_data.columns[[4, 5, 6, 7, 8, 9, 10]], axis=1)

# Apply the fix function FIRST
map_data['County FSA Code'] = map_data['County FSA Code'].apply(fix_fsa_code)

# THEN fix Pottawattamie counties AFTER the function
map_data.loc[map_data['County FSA Name'].str.contains('Pottawattamie', case=False, na=False), 'County FSA Code'] = '19155'
map_data.loc[map_data['County FSA Name'].str.contains('Pottawattamie', case=False, na=False), 'County FSA Name'] = 'Pottawattamie'

# Calculate crisis vs non-crisis disbursements by county
county_summary = map_data.groupby(['County FSA Code', 'County FSA Name', 'Crisis Payment'])['Disbursement Amount'].sum().reset_index()

# Pivot to get crisis and non-crisis amounts side by side
county_pivot = county_summary.pivot_table(
    index=['County FSA Code', 'County FSA Name'], 
    columns='Crisis Payment', 
    values='Disbursement Amount', 
    fill_value=0
).reset_index()

# Handle cases where not both 0 and 1 exist
if 0 not in county_pivot.columns:
    county_pivot[0] = 0
if 1 not in county_pivot.columns:
    county_pivot[1] = 0

# Rename columns for clarity
county_pivot.columns = ['County_FSA_Code', 'County_FSA_Name', 'Non_Crisis_Amount', 'Crisis_Amount']

# Convert to string for merging
county_pivot['County_FSA_Code'] = county_pivot['County_FSA_Code'].astype(str)

# Calculate totals
county_pivot['Total_Amount'] = county_pivot['Crisis_Amount'] + county_pivot['Non_Crisis_Amount']

# Download Iowa counties
url = "https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json"
counties = gpd.read_file(url)
iowa_counties = counties[counties['id'].str.startswith('19')]

# Merge with your data
iowa_counties = iowa_counties.merge(county_pivot, left_on='id', right_on='County_FSA_Code', how='left')

# Fix coordinate system
iowa_counties = iowa_counties.to_crs(epsg=4326)

# Create the map
fig, ax = plt.subplots(1, 1, figsize=(16, 12))

# Debug: Check how many counties should have data
print(f"Counties in data: {len(county_pivot)}")
print(f"Counties in map: {len(iowa_counties)}")
print(f"Counties with merged data: {len(iowa_counties[iowa_counties['Total_Amount'].notna()])}")

# Plot Iowa counties with borders FIRST
iowa_counties.plot(ax=ax, 
                   color='lightgray', 
                   edgecolor='black', 
                   linewidth=0.7)

# Get max amount for sizing
max_amount = county_pivot['Total_Amount'].max()

# Add pie charts using Wedge patches instead of ax.pie()
for idx, row in iowa_counties.iterrows():
    if pd.notna(row['Total_Amount']) and row['Total_Amount'] > 0:
        # Get county center
        centroid = row.geometry.centroid
        x, y = centroid.x, centroid.y
        
        # Calculate radius
        base_radius = 0.04
        size_factor = 0.1
        radius = base_radius + (row['Total_Amount'] / max_amount) * size_factor
        
        # Calculate proportions
        crisis_prop = row['Crisis_Amount'] / row['Total_Amount']
        crisis_angle = crisis_prop * 360
        
        # Create wedges manually
        # Non-crisis wedge (dark blue)
        wedge1 = Wedge((x, y), radius, 0, 360-crisis_angle, facecolor='darkblue', edgecolor='black', linewidth=0.5, hatch='///')
        ax.add_patch(wedge1)
        
        # Crisis wedge (orange)  
        if crisis_angle > 0:
            wedge2 = Wedge((x, y), radius, 360-crisis_angle, 360, facecolor='orange', edgecolor='black', linewidth=0.5)
            ax.add_patch(wedge2)

# Set title
ax.set_title('Iowa Counties: Crisis vs Non-Crisis Payment Disbursements', 
            fontsize=18, fontweight='bold')

# Add legend
legend_elements = [plt.Circle((0,0), 1, color='darkblue', label='Non-Crisis Payments'),
                  plt.Circle((0,0), 1, color='orange', label='Crisis Payments')]
ax.legend(handles=legend_elements, loc='upper right', fontsize = 14)

# Remove axes
ax.set_axis_off()

plt.tight_layout()
plt.show()

print(f"Counties with data: {len(county_pivot)}")
print(f"Total disbursements: ${county_pivot['Total_Amount'].sum():,.0f}")

##########################################
#                                        #
########### Export Short File ############
#                                        #
########################################## 

# Create short file for testing/development
# Need pip install openpyxl
fsa_short.to_csv('fsa_short.csv')