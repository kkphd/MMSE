# Predicting Cognitive Impairment by Kiran K.
# www.github.com/kkphd/mmse

# Download the longitudinal data frame from the Open Access Series of Imaging Studies (OASIS-2) project:
# https://www.kaggle.com/jboysen/mri-and-alzheimers

# Original data set website:
# http://www.oasis-brains.org/

# Acknowledgments:

# Data were provided [in part] by OASIS:
# Principal Investigators (Longitudinal): D. Marcus, R, Buckner, J. Csernansky, J. Morris;
# P50 AG05681, P01 AG03991, P01 AG026276, R01 AG021910, P20 MH071616, U24 RR021382

# AV-45 doses were provided by Avid Radiopharmaceuticals, a wholly owned subsidiary of Eli Lilly.

# Citation: https://doi.org/10.1162/jocn.2009.21407

import pandas as pd
import numpy as np
import seaborn as sns
import scipy


def import_analysis_df():
    analysis_df = pd.read_csv('Dataset/oasis.csv')

    # Brief overview of the data frame
    print(analysis_df.head(3))
    print(analysis_df.info())
    print(analysis_df.isnull().sum())


    # Clean the data set.
    analysis_df['Subject ID'] = analysis_df['Subject ID'].map(lambda x: x.lstrip('OAS'))
    analysis_df['Subject ID'] = analysis_df['Subject ID'].astype(np.int64)
    analysis_df.drop('Hand', axis=1, inplace=True)
    analysis_df.drop('SES', axis=1, inplace=True)
    analysis_df['MF_01'] = np.where(analysis_df['M/F'] == 'M', 0, 1)
    analysis_df = analysis_df.rename(columns={
        'Subject ID': 'ID', 'M/F': 'Sex', 'EDUC': 'Edu', 'MR Delay': 'MR_Delay'})
    analysis_df['CDR_Stage'] = analysis_df['CDR'].apply(cdr_to_stage)
    analysis_df['Age_Group'] = analysis_df['Age'].apply(age_num_to_group)
    analysis_df['Edu_Group'] = analysis_df['Edu'].apply(edu_to_group)
    analysis_columns = ['ID', 'Group', 'Visit', 'Sex', 'MF_01', 'Age', 'Age_Group',
                        'Edu', 'Edu_Group', 'MMSE', 'CDR', 'CDR_Stage', 'MR_Delay', 'eTIV', 'nWBV', 'ASF']
    analysis_df = analysis_df[analysis_columns]
    return analysis_df


def import_norms_df():
    norms_df = pd.read_excel('mmse.xlsx')
    norms_df['MMSE_Group'] = norms_df['MMSE_T'].apply(new_mmse_group)
    norms_df['MMSE_Percentile'] = norms_df['MMSE_T'].apply(t_to_percentile)
    return norms_df


def merged(analysis_df, norms_df):
    merged = pd.merge(analysis_df, norms_df, on=['MMSE', 'Age_Group', 'Edu_Group'], how='left')
    merged.rename(columns={'T': 'MMSE_T'})
    merged['MMSE_Label'] = merged['MMSE_Percentile'].apply(assign_label)
    merged['Group_Agreement'] = merged.apply(lambda row: agreement(row['Group'], row['MMSE_Group']), axis=1)
    merged['Group_Status'] = merged['Group'].apply(group_coding) # 0 = Cognitively intact; 1 = Cognitively impaired
    merged['MMSE_Group_Status'] = merged['MMSE_Group'].apply(group_coding)
    merged = merged[
        ['ID', 'Group', 'Group_Status', 'MMSE_Group', 'MMSE_Group_Status', 'Group_Agreement', 'Visit',
         'Sex', 'MF_01', 'Age', 'Age_Group', 'Edu', 'Edu_Group', 'MMSE', 'MMSE_T', 'MMSE_Percentile',
         'MMSE_Label', 'CDR', 'CDR_Stage', 'MR_Delay', 'eTIV', 'nWBV', 'ASF']]
    return merged


def run_analysis():
    analysis_df = import_analysis_df()
    norms_df = import_norms_df()
    results = merged(analysis_df, norms_df)
    return results


def cdr_to_stage(score):
    """
    Convert the Clinical Dementia Rating scale (CDR) to descriptive terms (O'Bryant et al., 2010).
    This can be helpful for qualitative purposes.

    """
    if score == 0.0:
        return ('Normal')
    elif score == 0.5:
            return ('Questionable')
    elif score == 1.0:
        return ('Mild')
    elif score == 2.0:
        return ('Moderate')
    elif score == 3.0:
        return ('Severe')
    else:
        return ('NaN')


def age_num_to_group(age):
    """
    A separate data file containing normative values for MMSE is ordered by age groups.
    This function will convert the interval value of age in the OASIS data set to
    age groups in order to find and convert the T-score more easily.
    """
    if 60 <= age <= 64:
        return "60-64"
    elif 65 <= age <= 69:
        return "65-69"
    elif 70 <= age <= 74:
        return "70-74"
    elif 75 <= age <= 79:
        return "75-79"
    elif 80 <= age <= 84:
        return "80-84"
    elif age >= 85:
        return ">=85"


def edu_to_group(edu):
    """
    A separate data file containing normative values for MMSE is also ordered by education.
    This function will convert the interval value of education in the OASIS data set to age groups
    in order to find and convert the T-score more easily.
    """
    if 0 <= edu <= 4:
        return "0-4"
    elif 5 <= edu <= 8:
        return "5-8"
    elif 9 <= edu <= 12:
        return "9-12"
    elif edu > 12:
        return ">12"


# Normative values for profoundly low performance (i.e., scores at or below T < 1) were converted to T = 1 for
# statistical purposes. Percentiles were calculated for each T score and assigned a descriptive label according to
# Guilmette et al. (2020) neuropsychological naming conventions.

# T scores <= 1.5 standard deviations relative to the mean were considered "Impaired" whereas
# scores > 1.5 were considered intact.

def new_mmse_group(mmse_t):
    """
    If the MMSE T-score is at or below 1.5 standard deviations from the mean, the participant is considered impaired, otherwise
    they are intact.
    """
    if mmse_t > 35:
        return "Intact"
    elif mmse_t <= 35:
        return "Impaired"


def group_coding(dementia_status):
    """
    Determines the level of agreement between the OASIS data set's classification of dementia versus the current
    study's (see function 'new_mmse_group').
    """
    if dementia_status == 'Nondemented':
        return 0
    elif dementia_status == 'Intact':
        return 0
    else:
        return 1


def agreement(group, mmse_group):
    if group == 'Nondemented' and mmse_group == 'Intact':
        return 1
    elif group == 'Nondemented' and mmse_group == 'Impaired':
        return 0
    elif group == 'Demented' and mmse_group == 'Intact':
        return 0
    elif group == 'Demented' and mmse_group == 'Impaired':
        return 1
    elif group == 'Converted' and mmse_group == 'Intact':
        return 0
    elif group == 'Converted' and mmse_group == 'Impaired':
        return 1


def t_to_percentile(t):
    return round(scipy.stats.norm.cdf(t, loc=50, scale=10)*100)


def assign_label(mmse_percentile):
    df = pd.read_csv('Dataset/oasis.csv')
    if df['MMSE'].skew() < 0:
        return skewed_percentile_to_label(mmse_percentile)
    else:
        return normal_percentile_to_label(mmse_percentile)


def skewed_percentile_to_label(percentile):
    """
    Assigns a descriptive term to the MMSE T-score based on its degree of skewness.
    """
    if percentile > 24:
        return 'WNL'
    elif 9 <= percentile <= 24:
        return 'Low Average'
    elif 2 <= percentile <= 8:
        return 'Below Average'
    elif percentile < 2:
        return 'Exceptionally Low'


def normal_percentile_to_label(percentile):
    """
    Assigns a descriptive term to the MMSE percentile score.
    """
    if percentile >= 98:
        return 'Exceptionally High'
    elif 91 <= percentile <= 97:
        return 'Above Average'
    elif 75 <= percentile <= 90:
        return 'High Average'
    elif 25 <= percentile <= 74:
        return 'Average'
    elif 9 <= percentile <= 24:
        return 'Low Average'
    elif 2 <= percentile <= 8:
        return 'Below Average'
    elif percentile < 2:
        return 'Exceptionally Low'



if __name__ == "__main__":
    analysis_results = run_analysis()