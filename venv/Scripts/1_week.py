import inline as inline
import matplotlib as matplotlib
import pandas as pd
import numpy as np
#%matplotlib inline
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

print(train_data)

df = pd.DataFrame([[1, 2], [4, 5], [7, 8]],
                  index=['cobra', 'viper', 'sidewinder'],
                  columns=['max_speed', 'shield'])

print(df)

tof = pd.Series([False, False, True],
                      index=['cobra', 'viper', 'sidewinder'],
                      dtype=bool)

print(tof)

print(df[tof])

print(f"{train_data.shape} {test_data.shape}")

noage_train_data = train_data[train_data['Age'].isna()]
havecabin_train_data  = train_data[~train_data['Cabin'].isna()]
knowage_train_data = train_data[~train_data['Age'].isna()]

figs, axes = plt.subplots(2, 2, figsize=(12, 13))

axes[0][0].boxplot(knowage_train_data['Age'])
axes[0][0].set_xlabel("Age")

axes[0][1].boxplot(train_data['Fare'])
axes[0][1].set_xlabel("Fare")

axes[1][0].hist(knowage_train_data['Age'], bins=10)
axes[1][0].set_xlabel("Age")

axes[1][1].hist(train_data['Fare'], bins=10)
axes[1][1].set_xlabel("Fare")

plt.show()

fig, axes = plt.subplots(1, 3, figsize=(25, 9))

sns.countplot(x='Sex', hue='Survived', data=train_data, ax=axes[0])
axes[0].set_title('Survival by sex')
axes[0].set_ylabel('')

sns.countplot(x='Pclass', hue='Survived', data=train_data, ax=axes[1])
axes[1].set_title('Survival by Pclass')
axes[1].set_ylabel('')

sns.countplot(x='Embarked', hue='Survived', data=train_data, ax=axes[2])
axes[2].set_title('Survival by Embarked')
axes[2].set_ylabel('')

plt.show()

plt.pie(train_data.groupby('Sex')[['Survived']].sum(),
        labels=['Female', 'Male'],
        autopct='%1.1f%%')
plt.title("Survival Ratio")

plt.show()

# train_data에서 'PassengerId'와 'Ticket'을 drop
train_data = train_data.drop(['PassengerId', 'Ticket'], axis=1)
train_data

# 예측값 파일 형식
sub_example = pd.read_csv('gender_submission.csv')
sub_example.head()

# test_data의 'PassengerId'는 Kaggle에서 예측값 평가 시 필요하기 때문에 'Ticket'만을 drop
test_data = test_data.drop(['Ticket'], axis=1)

train_data.dtypes

# 'SibSp'(형제자매와 배우자), 'Parch'(부모님과 자녀)를 합쳐 'Family'라는 새로운 feature를 만든다
train_data['Family'] = train_data['SibSp'] + train_data['Parch']
test_data['Family'] = test_data['SibSp'] + test_data['Parch']

# 가족의 수에 따른 평균 생존율
train_data.groupby('Family')[['Survived']].mean()

# 평균으로만 채우는 위의 방법이나 랜덤하게 채우는 아래의 방법으로 비어있는 'Age' 데이터를 채운다
# 다만 모든 가정은 머신러닝 모델이 배울 훈련 데이터를 기준으로 한다
def fill_age(df):
  age_avg = train_data['Age'].mean() # train_data의 'Age'의 평균을 사용
  age_std = train_data['Age'].std() # train_data의 'Age'의 표준편차를 사용
  age_nan_count = df['Age'].isna().sum()
  age_nan_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_nan_count)
  df['Age'][df['Age'].isna()] = age_nan_random_list
  df['Age'] = df['Age'].astype(int)

fill_age(train_data)
fill_age(test_data)

train_data.isna().sum(axis=0)
test_data.isna().sum(axis=0)

# train_data의 'Embarked'와 test_data의 'Fare' 데이터를 채운다
train_data["Embarked"].fillna('S', inplace=True)
test_data['Fare'].fillna(train_data['Fare'].median(), inplace=True)

# categorical 데이터인 성별을 numerical하게 바꾼다
train_data['Sex'] = train_data['Sex'].map({'male': 0, 'female': 1}).astype(int)
test_data['Sex'] = test_data['Sex'].map({'male': 0, 'female': 1}).astype(int)

# categorical한 'Embarked'도 바꿔준다
train_data['Embarked'] = train_data['Embarked'].replace('S', int(0))
train_data['Embarked'] = train_data['Embarked'].replace('C', int(1))
train_data['Embarked'] = train_data['Embarked'].replace('Q', int(2))

embarked_mapping = {'S': 0, 'C': 1, 'Q': 2}
test_data['Embarked'] = test_data['Embarked'].map(embarked_mapping).astype(int)

# 'Fare'를 binning을 하여 float을 int로 바꾼다
def bin_fare(df):
  df.loc[df['Fare'] <= 7.91, 'Fare'] = 0
  df.loc[(df['Fare'] > 7.91) & (df['Fare'] <= 14.454), 'Fare'] = 1
  df.loc[(df['Fare'] > 14.454) & (df['Fare'] <= 31), 'Fare'] = 2
  df.loc[df['Fare'] > 31, 'Fare'] =  3

  df['Fare'] = df['Fare'].astype(int)

# 'Age'도 binning을 통해 float을 int로 바꾼다
def bin_age(df):
  df.loc[df['Age'] <= 16, 'Age'] = 0
  df.loc[(df['Age'] > 16) & (df['Age'] <= 32), 'Age'] = 1
  df.loc[(df['Age'] > 32) & (df['Age'] <= 48), 'Age'] = 2
  df.loc[(df['Age'] > 48) & (df['Age'] <= 64), 'Age'] = 3
  df.loc[df['Age'] > 64, 'Age'] = 4

  df['Age'] = df['Age'].astype(int)

bin_fare(train_data)
bin_fare(test_data)
bin_age(train_data)
bin_age(test_data)

type(np.NaN)

# 'Cabin'의 유무로 생존율의 차이가 있기에, 'Has_Cabin'이라는 feature로 만들어준다
train_data['Has_Cabin'] = train_data['Cabin'].apply(lambda x: 0 if type(x) == float else 1)
test_data['Has_Cabin'] = test_data['Cabin'].apply(lambda x: 0 if type(x) == float else 1)

def get_title(name):
  for substring in name.split():
    if '.' in substring:
      return substring[:-1]

train_data['Title'] = train_data['Name'].apply(lambda x: get_title(x))
test_data['Title'] = test_data['Name'].apply(lambda x: get_title(x))

train_data['Title'].value_counts(dropna=False)
test_data['Title'].value_counts(dropna=False)

def map_title(df):
  df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
  df['Title'] = df['Title'].replace('Mlle', 'Miss')
  df['Title'] = df['Title'].replace('Ms', 'Miss')
  df['Title'] = df['Title'].replace('Mme', 'Mrs')
  title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
  df['Title'] = df['Title'].map(title_mapping)

map_title(train_data)
map_title(test_data)

train_data.drop(['Name', 'Cabin', 'SibSp', 'Parch'], axis=1, inplace = True)
test_data.drop(['Name', 'Cabin', 'SibSp', 'Parch'], axis=1, inplace = True)

train_data.info()
test_data.info()

print(train_data)
print(test_data)