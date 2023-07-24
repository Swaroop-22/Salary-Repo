import pandas as pd

#Loading of data set
df = pd.read_csv('glassdoor_jobs.csv')
print(df.head(5))
###############
#Salary Parsing
###############
df = df[df['Salary Estimate'] != '-1']   # Removing the salaries that having -1 values.
#There are few rows in salary estimate which have values like 1.Employer Provided Salary:$200K-$250K & 2.$24-$39 Per Hour(Glassdoor est.).
#Thus creating sepearte columns which contains 1 if there are Employer Provided Salary and Per Hour present else 0.
df['Hourly'] = df['Salary Estimate'].apply(lambda x: 1 if 'per hour' in x.lower() else 0)
df['Employer_provided'] = df['Salary Estimate'].apply(lambda x: 1 if 'Employer Provided Salary' in x.lower() else 0)
print(df['Hourly'].head(5))
print(df['Employer_provided'].head(5))
#Spliting the salary from i.e. removing the characters: $53K-$91K (Glassdoor est.) to 53-91
salary = df['Salary Estimate'].apply(lambda x: x.split('(')[0]) #$53K-$91K (Glassdoor est.) - $53K-$91K.
print(salary)
minus_KD = salary.apply(lambda x: x.replace('$', '').replace('K', '')) #Replacing the $ and K with blank, thus 53-91.
print(minus_KD)

min_hr = minus_KD.apply(lambda x: x.lower().replace('per hour', '').replace('employer provided salary:', ''))
print(min_hr)
#Creating new columns for Minimum, Max and Avg for salary.
df['min_salary'] = min_hr.apply(lambda x: int(x.split('-')[0]))# 53-91, thus 53.
df['max_salary'] = min_hr.apply(lambda x: int(x.split('-')[1]))# 53-91, thus 91.
df['avg_salary'] = (df.min_salary + df.max_salary) / 2 # 53-91, thus 98.5
print(df['min_salary'].head(5))
print(df['max_salary'].head(5))
print(df['avg_salary'].head(5))

####################################
#Comapny name Text only
####################################
df['company_text'] = df.apply(lambda x: x['Company Name'] if x['Rating'] < 0 else x['Company Name'][:-3 ], axis = 1)
df['company_text'] = df.company_text.apply(lambda x: x.replace('\n', '')) #removing new line from company text column.
print(df['company_text'].head(5))

####################################
#State Field
####################################
df['job_state'] = df['Location'].apply(lambda x: x.split(',')[1])
print(df.job_state.value_counts())
df['job_state'] = df.job_state.apply(lambda x: x.strip() if x.strip().lower() != 'los angeles' else 'CA')
print(df.job_state.value_counts())
print(df['job_state'].head(5))

#Checking wether the job location and headquarters are in same location.
df['same_state'] = df.apply(lambda x: 1 if x.Location == x.Headquarters else 0, axis =1)
print(df['same_state'].head(5))

#####################################
#Age of Company
#####################################
df['age'] = df.Founded.apply(lambda x: x if x < 1 else 2020 - x)
print(df['age'].head(5))

#####################################
#Job Description
#####################################
#Related details: 1. Python, 2. R-studio 3. AWS etc....
df['python'] = df['Job Description'].apply(lambda x: 1 if 'python' in x.lower() else 0)
print(df.python.value_counts())
#R-studio
df['R_studio'] = df['Job Description'].apply(lambda x: 1 if 'r-studio' in x.lower() or 'r studio' in x.lower() else 0)
print(df.R_studio.value_counts())
#Spark
df['Spark'] = df['Job Description'].apply(lambda x: 1 if 'spark' in x.lower() else 0)
print(df.Spark.value_counts())
#AWS
df['AWS'] = df['Job Description'].apply(lambda x: 1 if 'aws' in x.lower() else 0)
print(df.AWS.value_counts())

df['desc_len'] = df['Job Description'].apply(lambda x: len(x))
print(df['desc_len'].head(5))

#########################################
# Number of Competitors
#########################################
df['num_of_comp'] = df['Competitors'].apply(lambda x:len(x.split(',')) if x != '-1' else 0)
print(df['num_of_comp'].head(5))

#########################################
#Hourly to annual Wages
#########################################
df['min_salary'] = df.apply(lambda x: x.min_salary*2 if x.Hourly == 1 else x.min_salary, axis = 1)
df['max_salary'] = df.apply(lambda x: x.max_salary*2 if x.Hourly == 1 else x.max_salary, axis = 1)
print(df[df.Hourly == 1][['Hourly', 'min_salary', 'max_salary']])

#########################################
#Simplifying Job Titles
#########################################
def title_simplifier(title):
    if 'data scientist' in title.lower():
        return 'data scientist'
    elif 'data engineer' in title.lower():
        return 'data engineer'
    elif 'analyst' in title.lower():
        return 'analyst'
    elif 'machine learning' in title.lower():
        return 'mle'
    elif 'manager' in title.lower():
        return 'manager'
    elif 'director' in title.lower():
        return 'director'
    else:
        return 'na'

def seniority(title):
    if 'sr' in title.lower() or 'senior' in title.lower() or 'lead' in title.lower() or 'principal' in title.lower():
        return 'senior'
    elif 'jr' in title.lower() or 'jr.' in title.lower():
        return 'junior'
    else:
        return 'na'


df['job_simp'] = df['Job Title'].apply(title_simplifier)
print(df.job_simp.value_counts())

df['seniority'] = df['Job Title'].apply(seniority)
print(df.seniority.value_counts())

#########################################
#Dropping Unwanted Columns
#########################################
print(df.columns)
data = df.drop(['Unnamed: 0'], axis = 1)
data.to_csv('salary_data_cleaned.csv', index = False)
