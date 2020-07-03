# Predicting Cognitive Impairment by Kiran K.
# www.github.com/kkphd/mmse


import Step_1_MMSE_Data_Cleaning
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import pprint

analysis_results = Step_1_MMSE_Data_Cleaning.run_analysis()


# Perform exploratory data analysis by first filtering for visit 1 cases ("v1") and sub-setting the groups.
# Group membership is based on their cognitive test score.
v1 = analysis_results['Visit'] == 1
v1_all = analysis_results[v1]
has_unique_cases = v1_all.duplicated('ID').sum() == 0
is_null = v1_all.isnull().sum()

v1_all_nrow = len(v1_all)
v1_all_ncol = len(v1_all.columns)
intact = v1_all['MMSE_Group'] == 'Intact'
v1_intact = v1_all[intact]
impaired = v1_all['MMSE_Group'] == 'Impaired'
v1_impaired = v1_all[impaired]


# Determine value counts of Intact and Impaired individuals.
v1_all['MMSE_Group'].value_counts()
# Intact: N = 124
# Impaired: N = 26

# Determine the types of variables we are working with.
v1_all.info()


# Exploratory data analysis:
group_agree = v1_all['Group_Agreement'].mean()*100


def v1(v1_all):
    v1_desc = v1_all.groupby('MMSE_Group').agg(
        {
            'MF_01': ['count', 'mean', 'std', 'min', 'max', 'median'],
            'Age': ['count', 'mean', 'std', 'min', 'max', 'median', 'skew'],
            'Edu': ['count', 'mean', 'std', 'min', 'max', 'median', 'skew'],
            'MMSE': ['count', 'mean', 'std', 'min', 'max', 'median', 'skew'],
            'MMSE_T': ['count', 'mean', 'std', 'min', 'max', 'median', 'skew'],
            'MMSE_Percentile': ['count', 'mean', 'std', 'min', 'max', 'median', 'skew'],
            'CDR': ['count', 'mean', 'std', 'min', 'max', 'median'],
            'eTIV': ['count', 'mean', 'std', 'min', 'max', 'median', 'skew'],
            'nWBV': ['count', 'mean', 'std', 'min', 'max', 'median', 'skew'],
            'ASF': ['count', 'mean', 'std', 'min', 'max', 'median', 'skew']
        }
    )
    return np.transpose(v1_desc)


def create_group_plots():
    fig1, ax1 = plt.subplots(2, 2, figsize=(16, 10))
    ax1[0, 0].set_title('Distribution of Age by Group', size='14')
    ax1[0, 1].set_title('Distribution of Education by Group', size='14')
    ax1[1, 0].set_title('Distribution of Cognitive Status', size='14')
    ax1[1, 1].set_title('Distribution of CDR Stage', size='14')
    fig1.subplots_adjust(hspace=0.3)
    sns.set(style='whitegrid', palette='muted')
    age_order = ['60-64', '65-69', '70-74', '75-79', '80-84', '>=85']
    edu_order = ['5-8', '9-12', '>12']
    age_fig = sns.countplot(data=v1_all, x='Age_Group', hue='MMSE_Group', ax=ax1[0, 0], order=age_order)
    edu_fig = sns.countplot(data=v1_all, x='Edu_Group', hue='MMSE_Group', ax=ax1[0, 1], order=edu_order)
    dem_fig = sns.countplot(data=v1_all, x='MMSE_Label', ax=ax1[1, 0])
    cdr_fig = sns.countplot(data=v1_all, x='CDR_Stage', ax=ax1[1, 1])
    age_fig.set_xlabel('Age Group')
    edu_fig.set_xlabel('Years of Education')
    dem_fig.set_xlabel('Level of Cognitive Impairment')
    cdr_fig.set_xlabel('CDR Stage')

create_group_plots()


# The data suggests we have an imbalanced sample favoring normals.
def freq_figures():
    fig2, ax2 = plt.subplots(1, 2, figsize=(12, 6))
    fig2.suptitle('Number of Participants per Group', size='16')
    sns.set(style='whitegrid', palette='muted')
    sns.countplot(data=v1_all, x='Group', ax=ax2[0])
    sns.countplot(data=v1_all, x='MMSE_Group', ax=ax2[1])

freq_figures()


if has_unique_cases:
    v1_eda = v1(v1_all)
    v1_eda = v1_eda.rename(columns={'MMSE_Group': 'Group'},
                           index={'MF_01': 'Sex', 'Edu': 'Education (years)', 'MMSE_T': 'MMSE T Score'})
    print(v1_eda)

pprint.pprint(v1_eda)
# Variables were considered skewed if it was ~ < -1 or ~ > 1.
# Intact: MMSE, MMSE_Percentile, and eTIV were moderately skewed.
# Impaired: Education was moderately skewed.


# Test for normality of continuous variables by performing the Shapiro-Wilk test.
def calc_shapiro():
    shapiro_all = {}
    shapiro_intact = {}
    shapiro_impaired = {}
    shapiro_columns = ['Age', 'Edu', 'MMSE', 'MMSE_T', 'MMSE_Percentile', 'eTIV', 'nWBV', 'ASF']
    for column in shapiro_columns:
        shapiro_all[column] = stats.shapiro(v1_intact[column])
        shapiro_intact[column] = stats.shapiro(v1_intact[column])
        shapiro_impaired[column] = stats.shapiro(v1_impaired[column])
    return shapiro_all, shapiro_intact, shapiro_impaired

shapiro_all, shapiro_intact, shapiro_impaired = calc_shapiro()
# Intact group: Edu, MMSE, MMSE_T, MMSE_Percentile, and eTIV were not derived from a normal distribution.
# Impaired group: Edu, MMSE_T, and MMSE_Percentile were not derived from a normal distribution.


shapiro_all_df = pd.DataFrame.from_dict(shapiro_all)
shapiro_all_df = pd.DataFrame(shapiro_all, index=['Intact', 'Impaired'])
shapiro_intact_df = pd.DataFrame.from_dict(shapiro_intact)
shapiro_impaired_df = pd.DataFrame.from_dict(shapiro_impaired)


# Test for kurtosis.
def calc_kurt():
    kurt_all = {}
    kurt_intact = {}
    kurt_impaired = {}
    kurt_columns = ['MF_01', 'Age', 'Edu', 'MMSE', 'MMSE_T', 'MMSE_Percentile', 'CDR', 'eTIV', 'nWBV', 'ASF']
    for column in kurt_columns:
        kurt_all[column] = stats.kurtosis(v1_all[column])
        kurt_intact[column] = stats.kurtosis(v1_intact[column])
        kurt_impaired[column] = stats.kurtosis(v1_impaired[column])
    return kurt_all, kurt_intact, kurt_impaired

kurt_all, kurt_intact, kurt_impaired = calc_kurt()
# The distributions are largely platykurtic.


kurt_all_df = pd.DataFrame(kurt_all)
kurt_intact_df = pd.DataFrame(kurt_intact, index=[0])
kurt_impaired_df = pd.DataFrame(kurt_impaired, index=[1])


def mmse_figure():
    plt.plot(figsize=(15, 8))
    plt.figure(3)
    plt.suptitle('MMSE Raw Score by Group')
    sns.set_style('whitegrid')
    sns.boxplot(x=v1_all['MMSE_Group'], y=v1_all['MMSE'], palette='RdBu_r')
    sns.stripplot(x=v1_all['MMSE_Group'], y=v1_all['MMSE'], jitter=True, marker='D', color='black')
    sns.despine()
    plt.xlabel('Group')
    plt.ylabel('MMSE Raw Score')
    plt.ylim(15, 31)

mmse_figure()


# Create a Spearman correlation matrix due to evidence of violated assumptions of normality.
def v1_spearman_corr():
    v1_continuous = v1_all.drop(columns=['ID', 'Visit', 'Group', 'MMSE_Group', 'Age_Group',
                                         'Edu_Group', 'MMSE_Label', 'CDR_Stage', 'MR_Delay', 'ASF'])
    axes_labels = ['Group Dementia Status', 'MMSE Dementia Status', 'Status Agreement', 'Sex', 'Age',
                   'Education', 'MMSE', 'MMSE T-Score', 'MMSE %ile', 'CDR', 'eTIV', 'nWBV']
    corr_matrix = v1_continuous.corr(method='spearman')
    fig, ax = plt.subplots(figsize=(13, 10))
    corr_heatmap = sns.heatmap(corr_matrix, annot=True)
    corr_heatmap.set_xticklabels(axes_labels, rotation=45)
    corr_heatmap.set_yticklabels(axes_labels)
    plt.tight_layout()
    return corr_matrix, corr_heatmap

v1_corr_map = v1_spearman_corr()
spearman_figure = plt.title('Spearman Correlation Heatmap', fontsize=15)


# Perform nonparametric Spearman rank correlation significance testing.
def calc_spearman():
    columns = ['MF_01', 'Age', 'Edu', 'MMSE', 'MMSE_T', 'MMSE_Percentile', 'CDR', 'eTIV', 'nWBV']
    spearman_pvalues = {}
      # loop through each column
    for column in columns:
        spear_arr = [] # setup an empty array to load up with data that will assign to dictionary
        for col in columns: # loop through each column again with a different name
            correlation, p_value = stats.spearmanr(v1_all[column], v1_all[col]) # get your values from stat
            spear_arr.append({col: (correlation, p_value)}) # assign your values to dictionary as needed
        spearman_pvalues[column] = spear_arr # add the array to the dictionary once we've looped through second loop

        # spearman_pvalues[column] = stats.spearmanr(v1_all[column][0], v1_all[column][1])
        # spearman_intact_pvalues[column] = stats.spearmanr(v1_intact[column][0], v1_intact[column][1])
        # spearman_impaired_pvalues[column] = stats.spearmanr(v1_impaired[column][0], v1_impaired[column][1])
    return spearman_pvalues
        #, spearman_intact_pvalues, spearman_impaired_pvalues

spearman_results = calc_spearman()


spearman_all_dt = pd.DataFrame.from_dict(spearman_results)


# Calculate Spearman correlation scores for the Intact group.
def calc_intact_spearman():
    columns = ['MF_01', 'Age', 'Edu', 'MMSE', 'MMSE_T', 'MMSE_Percentile', 'CDR', 'eTIV', 'nWBV']
    spearman_intact_pvalues = {}
    for column in columns:
        spear_arr = []
        for col in columns:
            correlation, p_value = stats.spearmanr(v1_intact[column], v1_intact[col])
            spear_arr.append({col: (correlation, p_value)})
        spearman_intact_pvalues[column] = spear_arr
    return spearman_intact_pvalues

spearman_intact_results = calc_intact_spearman()


# Calculate Spearman correlation scores for the Impaired group.
def calc_impaired_spearman():
    columns = ['MF_01', 'Age', 'Edu', 'MMSE', 'MMSE_T', 'MMSE_Percentile', 'CDR', 'eTIV', 'nWBV']
    spearman_impaired_pvalues = {}
    for column in columns:
        spear_arr = []
        for col in columns:
            correlation, p_value = stats.spearmanr(v1_impaired[column], v1_impaired[col])
            spear_arr.append({col: (correlation, p_value)})
        spearman_impaired_pvalues[column] = spear_arr
    return spearman_impaired_pvalues
        #, spearman_intact_pvalues, spearman_impaired_pvalues

spearman_impaired_results = calc_impaired_spearman()


# Assess for homogeniety of variance using median-based Levene's tests.
def calc_levene():
    levene = {}
    levene_columns = ['Age', 'Edu', 'MMSE', 'MMSE_T', 'MMSE_Percentile', 'eTIV', 'nWBV', 'ASF']
    for column in levene_columns:
        levene[column] = stats.levene(v1_intact[column], v1_impaired[column])
    return levene

levene = calc_levene()


# Since there was some evidence of unequal variance between the two groups - given by the MMSE values -
# assess for differences between normals and participants with cognitive impairment by using Welch's t-test.
# Even if homogeneity of variance assumptions were not violated, this approach is still sometimes
# preferred over t-tests (Delacre, Lakens, & Leys, 2017).
def calc_welch():
    welch = {}
    welch_columns = ['MF_01', 'Age', 'Edu', 'MMSE', 'MMSE_T', 'CDR', 'eTIV', 'nWBV', 'ASF']
    for column in welch_columns:
        welch[column] = stats.ttest_ind(v1_intact[column], v1_impaired[column], equal_var=False)
    return welch

welch_results = calc_welch()


# Plot pairwise variable relationships.
def plot_dist():
    sns.set(style='whitegrid', color_codes=True)
    v1_interval = v1_all[['Age', 'MMSE_Group', 'Edu', 'MMSE', 'MMSE_T', 'MMSE_Percentile',
                          'eTIV', 'nWBV']]
    f = sns.pairplot(v1_interval, hue='MMSE_Group')
    f.add_legend()
    f.fig.set_figheight(18)
    f.fig.set_figwidth(22)
    f.fig.suptitle('Distribution of Age, Education, Cognitive Status, and Neuroimaging Factors'
                   ' by Cognitive Status', size='16')
    plt.subplots_adjust(top=0.95)

plot_dist()