import numpy as np
import pandas as pd
import warnings
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from wordcloud import WordCloud

warnings.filterwarnings("ignore")
import os
import seaborn as sns  # 数据模块可视化

os.chdir(r'C:\Users\ASUS\Desktop\coursera\movie\data')

def prepare(df):
    global json_cols
    global train_dict
    df['rating'] = df['rating'].fillna(1.5)
    df['totalVotes'] = df['totalVotes'].fillna(6)
    df['weightedRating'] = (df['rating'] * df['totalVotes'] + 6.367 * 300) / (df['totalVotes'] + 300)

    df[['release_month', 'release_day', 'release_year']] = df['release_date'].str.split('/', expand=True).replace(
        np.nan, 0).astype(int)
    df['release_year'] = df['release_year']
    df.loc[(df['release_year'] <= 19) & (df['release_year'] < 100), "release_year"] += 2000
    df.loc[(df['release_year'] > 19) & (df['release_year'] < 100), "release_year"] += 1900

    releaseDate = pd.to_datetime(df['release_date'])
    df['release_dayofweek'] = releaseDate.dt.dayofweek
    df['release_quarter'] = releaseDate.dt.quarter

    df['originalBudget'] = df['budget']
    df['inflationBudget'] = df['budget'] + df['budget'] * 1.8 / 100 * (
            2018 - df['release_year'])  # Inflation simple formula
    df['budget'] = np.log1p(df['budget'])

    df['genders_0_crew'] = df['crew'].apply(lambda x: sum([1 for i in x if i['gender'] == 0]))
    df['genders_1_crew'] = df['crew'].apply(lambda x: sum([1 for i in x if i['gender'] == 1]))
    df['genders_2_crew'] = df['crew'].apply(lambda x: sum([1 for i in x if i['gender'] == 2]))
    df['_collection_name'] = df['belongs_to_collection'].apply(lambda x: x[0]['name'] if x != {} else '').fillna('')
    le = LabelEncoder()
    df['_collection_name'] = le.fit_transform(df['_collection_name'])
    df['_num_Keywords'] = df['Keywords'].apply(lambda x: len(x) if x != {} else 0)
    df['_num_cast'] = df['cast'].apply(lambda x: len(x) if x != {} else 0)

    df['_popularity_mean_year'] = df['popularity'] / df.groupby("release_year")["popularity"].transform('mean')
    df['_budget_runtime_ratio'] = df['budget'] / df['runtime']
    df['_budget_popularity_ratio'] = df['budget'] / df['popularity']
    df['_budget_year_ratio'] = df['budget'] / (df['release_year'] * df['release_year'])
    df['_releaseYear_popularity_ratio'] = df['release_year'] / df['popularity']
    df['_releaseYear_popularity_ratio2'] = df['popularity'] / df['release_year']

    df['_popularity_totalVotes_ratio'] = df['totalVotes'] / df['popularity']
    df['_rating_popularity_ratio'] = df['rating'] / df['popularity']
    df['_rating_totalVotes_ratio'] = df['totalVotes'] / df['rating']
    df['_totalVotes_releaseYear_ratio'] = df['totalVotes'] / df['release_year']
    df['_budget_rating_ratio'] = df['budget'] / df['rating']
    df['_runtime_rating_ratio'] = df['runtime'] / df['rating']
    df['_budget_totalVotes_ratio'] = df['budget'] / df['totalVotes']

    df['has_homepage'] = 1
    df.loc[pd.isnull(df['homepage']), "has_homepage"] = 0

    df['isbelongs_to_collectionNA'] = 0
    df.loc[pd.isnull(df['belongs_to_collection']), "isbelongs_to_collectionNA"] = 1

    df['isTaglineNA'] = 0
    df.loc[df['tagline'] == 0, "isTaglineNA"] = 1

    df['isOriginalLanguageEng'] = 0
    df.loc[df['original_language'] == "en", "isOriginalLanguageEng"] = 1

    df['isTitleDifferent'] = 1
    df.loc[df['original_title'] == df['title'], "isTitleDifferent"] = 0

    df['isMovieReleased'] = 1
    df.loc[df['status'] != "Released", "isMovieReleased"] = 0

    # get collection id
    df['collection_id'] = df['belongs_to_collection'].apply(lambda x: np.nan if len(x) == 0 else x[0]['id'])

    df['original_title_letter_count'] = df['original_title'].str.len()
    df['original_title_word_count'] = df['original_title'].str.split().str.len()

    df['title_word_count'] = df['title'].str.split().str.len()
    df['overview_word_count'] = df['overview'].str.split().str.len()
    df['tagline_word_count'] = df['tagline'].str.split().str.len()

    df['production_countries_count'] = df['production_countries'].apply(lambda x: len(x))
    df['production_companies_count'] = df['production_companies'].apply(lambda x: len(x))
    df['crew_count'] = df['crew'].apply(lambda x: len(x) if x != {} else 0)

    for col in ['genres', 'production_countries', 'spoken_languages', 'production_companies']:
        df[col] = df[col].map(lambda x: sorted(
            list(set([n if n in train_dict[col] else col + '_etc' for n in [d['name'] for d in x]])))).map(
            lambda x: ','.join(map(str, x)))
        temp = df[col].str.get_dummies(sep=',')
        df = pd.concat([df, temp], axis=1, sort=False)
    df.drop(['genres_etc'], axis=1, inplace=True)

    df = df.drop(['id', 'revenue', 'belongs_to_collection', 'genres', 'homepage', 'imdb_id', 'overview', 'runtime'
                     , 'poster_path', 'production_companies', 'production_countries', 'release_date', 'spoken_languages'
                     , 'status', 'title', 'Keywords', 'cast', 'crew', 'original_language', 'original_title', 'tagline',
                  'collection_id'
                  ], axis=1)

    df.fillna(value=0.0, inplace=True)
    return df


def get_dictionary(s):
    try:
        d = eval(s)
    except:
        d = {}
    return d


def get_json_dict(df):
    global json_cols
    result = dict()
    for e_col in json_cols:
        d = dict()
        rows = df[e_col].values
        for row in rows:
            if row is None: continue
            for i in row:
                if i['name'] not in d:
                    d[i['name']] = 0
                d[i['name']] += 1
        result[e_col] = d
    return result


if __name__ == '__main__':
    train = pd.read_csv('C:\Projects\old\电影数据集多模型对比\data/train.csv')
    train.loc[train['id'] == 16, 'revenue'] = 192864  # Skinning
    train.loc[train['id'] == 90, 'budget'] = 30000000  # Sommersby
    train.loc[train['id'] == 118, 'budget'] = 60000000  # Wild Hogs
    train.loc[train['id'] == 149, 'budget'] = 18000000  # Beethoven
    train.loc[train['id'] == 313, 'revenue'] = 12000000  # The Cookout
    train.loc[train['id'] == 451, 'revenue'] = 12000000  # Chasing Liberty
    train.loc[train['id'] == 464, 'budget'] = 20000000  # Parenthood
    train.loc[train['id'] == 470, 'budget'] = 13000000  # The Karate Kid, Part II
    train.loc[train['id'] == 513, 'budget'] = 930000  # From Prada to Nada
    train.loc[train['id'] == 797, 'budget'] = 8000000  # Welcome to Dongmakgol
    train.loc[train['id'] == 819, 'budget'] = 90000000  # Alvin and the Chipmunks: The Road Chip
    train.loc[train['id'] == 850, 'budget'] = 90000000  # Modern Times
    train.loc[train['id'] == 1007, 'budget'] = 2  # Zyzzyx Road
    train.loc[train['id'] == 1112, 'budget'] = 7500000  # An Officer and a Gentleman
    train.loc[train['id'] == 1131, 'budget'] = 4300000  # Smokey and the Bandit
    train.loc[train['id'] == 1359, 'budget'] = 10000000  # Stir Crazy
    train.loc[train['id'] == 1542, 'budget'] = 1  # All at Once
    train.loc[train['id'] == 1570, 'budget'] = 15800000  # Crocodile Dundee II
    train.loc[train['id'] == 1571, 'budget'] = 4000000  # Lady and the Tramp
    train.loc[train['id'] == 1714, 'budget'] = 46000000  # The Recruit
    train.loc[train['id'] == 1721, 'budget'] = 17500000  # Cocoon
    train.loc[train['id'] == 1865, 'revenue'] = 25000000  # Scooby-Doo 2: Monsters Unleashed
    train.loc[train['id'] == 1885, 'budget'] = 12  # In the Cut
    train.loc[train['id'] == 2091, 'budget'] = 10  # Deadfall
    train.loc[train['id'] == 2268, 'budget'] = 17500000  # Madea Goes to Jail budget
    train.loc[train['id'] == 2491, 'budget'] = 6  # Never Talk to Strangers
    train.loc[train['id'] == 2602, 'budget'] = 31000000  # Mr. Holland's Opus
    train.loc[train['id'] == 2612, 'budget'] = 15000000  # Field of Dreams
    train.loc[train['id'] == 2696, 'budget'] = 10000000  # Nurse 3-D
    train.loc[train['id'] == 2801, 'budget'] = 10000000  # Fracture
    train.loc[train['id'] == 335, 'budget'] = 2
    train.loc[train['id'] == 348, 'budget'] = 12
    train.loc[train['id'] == 470, 'budget'] = 13000000
    train.loc[train['id'] == 513, 'budget'] = 1100000
    train.loc[train['id'] == 640, 'budget'] = 6
    train.loc[train['id'] == 696, 'budget'] = 1
    train.loc[train['id'] == 797, 'budget'] = 8000000
    train.loc[train['id'] == 850, 'budget'] = 1500000
    train.loc[train['id'] == 1199, 'budget'] = 5
    train.loc[train['id'] == 1282, 'budget'] = 9  # Death at a Funeral
    train.loc[train['id'] == 1347, 'budget'] = 1
    train.loc[train['id'] == 1755, 'budget'] = 2
    train.loc[train['id'] == 1801, 'budget'] = 5
    train.loc[train['id'] == 1918, 'budget'] = 592
    train.loc[train['id'] == 2033, 'budget'] = 4
    train.loc[train['id'] == 2118, 'budget'] = 344
    train.loc[train['id'] == 2252, 'budget'] = 130
    train.loc[train['id'] == 2256, 'budget'] = 1
    train.loc[train['id'] == 2696, 'budget'] = 10000000

    test = pd.read_csv('C:\Projects\old\电影数据集多模型对比\data/test.csv')

    # Clean Data
    test.loc[test['id'] == 6733, 'budget'] = 5000000
    test.loc[test['id'] == 3889, 'budget'] = 15000000
    test.loc[test['id'] == 6683, 'budget'] = 50000000
    test.loc[test['id'] == 5704, 'budget'] = 4300000
    test.loc[test['id'] == 6109, 'budget'] = 281756
    test.loc[test['id'] == 7242, 'budget'] = 10000000
    test.loc[test['id'] == 7021, 'budget'] = 17540562  # Two Is a Family
    test.loc[test['id'] == 5591, 'budget'] = 4000000  # The Orphanage
    test.loc[test['id'] == 4282, 'budget'] = 20000000  # Big Top Pee-wee
    test.loc[test['id'] == 3033, 'budget'] = 250
    test.loc[test['id'] == 3051, 'budget'] = 50
    test.loc[test['id'] == 3084, 'budget'] = 337
    test.loc[test['id'] == 3224, 'budget'] = 4
    test.loc[test['id'] == 3594, 'budget'] = 25
    test.loc[test['id'] == 3619, 'budget'] = 500
    test.loc[test['id'] == 3831, 'budget'] = 3
    test.loc[test['id'] == 3935, 'budget'] = 500
    test.loc[test['id'] == 4049, 'budget'] = 995946
    test.loc[test['id'] == 4424, 'budget'] = 3
    test.loc[test['id'] == 4460, 'budget'] = 8
    test.loc[test['id'] == 4555, 'budget'] = 1200000
    test.loc[test['id'] == 4624, 'budget'] = 30
    test.loc[test['id'] == 4645, 'budget'] = 500
    test.loc[test['id'] == 4709, 'budget'] = 450
    test.loc[test['id'] == 4839, 'budget'] = 7
    test.loc[test['id'] == 3125, 'budget'] = 25
    test.loc[test['id'] == 3142, 'budget'] = 1
    test.loc[test['id'] == 3201, 'budget'] = 450
    test.loc[test['id'] == 3222, 'budget'] = 6
    test.loc[test['id'] == 3545, 'budget'] = 38
    test.loc[test['id'] == 3670, 'budget'] = 18
    test.loc[test['id'] == 3792, 'budget'] = 19
    test.loc[test['id'] == 3881, 'budget'] = 7
    test.loc[test['id'] == 3969, 'budget'] = 400
    test.loc[test['id'] == 4196, 'budget'] = 6
    test.loc[test['id'] == 4221, 'budget'] = 11
    test.loc[test['id'] == 4222, 'budget'] = 500
    test.loc[test['id'] == 4285, 'budget'] = 11
    test.loc[test['id'] == 4319, 'budget'] = 1
    test.loc[test['id'] == 4639, 'budget'] = 10
    test.loc[test['id'] == 4719, 'budget'] = 45
    test.loc[test['id'] == 4822, 'budget'] = 22
    test.loc[test['id'] == 4829, 'budget'] = 20
    test.loc[test['id'] == 4969, 'budget'] = 20
    test.loc[test['id'] == 5021, 'budget'] = 40
    test.loc[test['id'] == 5035, 'budget'] = 1
    test.loc[test['id'] == 5063, 'budget'] = 14
    test.loc[test['id'] == 5119, 'budget'] = 2
    test.loc[test['id'] == 5214, 'budget'] = 30
    test.loc[test['id'] == 5221, 'budget'] = 50
    test.loc[test['id'] == 4903, 'budget'] = 15
    test.loc[test['id'] == 4983, 'budget'] = 3
    test.loc[test['id'] == 5102, 'budget'] = 28
    test.loc[test['id'] == 5217, 'budget'] = 75
    test.loc[test['id'] == 5224, 'budget'] = 3
    test.loc[test['id'] == 5469, 'budget'] = 20
    test.loc[test['id'] == 5840, 'budget'] = 1
    test.loc[test['id'] == 5960, 'budget'] = 30
    test.loc[test['id'] == 6506, 'budget'] = 11
    test.loc[test['id'] == 6553, 'budget'] = 280
    test.loc[test['id'] == 6561, 'budget'] = 7
    test.loc[test['id'] == 6582, 'budget'] = 218
    test.loc[test['id'] == 6638, 'budget'] = 5
    test.loc[test['id'] == 6749, 'budget'] = 8
    test.loc[test['id'] == 6759, 'budget'] = 50
    test.loc[test['id'] == 6856, 'budget'] = 10
    test.loc[test['id'] == 6858, 'budget'] = 100
    test.loc[test['id'] == 6876, 'budget'] = 250
    test.loc[test['id'] == 6972, 'budget'] = 1
    test.loc[test['id'] == 7079, 'budget'] = 8000000
    test.loc[test['id'] == 7150, 'budget'] = 118
    test.loc[test['id'] == 6506, 'budget'] = 118
    test.loc[test['id'] == 7225, 'budget'] = 6
    test.loc[test['id'] == 7231, 'budget'] = 85
    test.loc[test['id'] == 5222, 'budget'] = 5
    test.loc[test['id'] == 5322, 'budget'] = 90
    test.loc[test['id'] == 5350, 'budget'] = 70
    test.loc[test['id'] == 5378, 'budget'] = 10
    test.loc[test['id'] == 5545, 'budget'] = 80
    test.loc[test['id'] == 5810, 'budget'] = 8
    test.loc[test['id'] == 5926, 'budget'] = 300
    test.loc[test['id'] == 5927, 'budget'] = 4
    test.loc[test['id'] == 5986, 'budget'] = 1
    test.loc[test['id'] == 6053, 'budget'] = 20
    test.loc[test['id'] == 6104, 'budget'] = 1
    test.loc[test['id'] == 6130, 'budget'] = 30
    test.loc[test['id'] == 6301, 'budget'] = 150
    test.loc[test['id'] == 6276, 'budget'] = 100
    test.loc[test['id'] == 6473, 'budget'] = 100
    test.loc[test['id'] == 6842, 'budget'] = 30

    # features from https://www.kaggle.com/kamalchhirang/eda-simple-feature-engineering-external-data
    train = pd.merge(train, pd.read_csv('./TrainAdditionalFeatures.csv'),
                     how='left', on=['imdb_id'])
    test = pd.merge(test, pd.read_csv('./TestAdditionalFeatures.csv'),
                    how='left', on=['imdb_id'])

    additionalTrainData = pd.read_csv('./additionalTrainData.csv')
    additionalTrainData['release_date'] = additionalTrainData['release_date'].astype('str').str.replace('-', '/')
    train = pd.concat([train, additionalTrainData])
    print(train.columns)
    print(train.shape)
    train['revenue'] = np.log1p(train['revenue'])
    train['log_revenue'] = np.log1p(train['revenue'])
    y = train['revenue']
    # 画收入原始图和np.log1p后的图：票房分配有很高的偏差！ 最好使用np.log1p的收入
    fig, ax = plt.subplots(figsize=(16, 6))
    plt.subplot(1, 2, 1)
    plt.hist(train['revenue'])
    plt.title('Distribution of revenue')
    plt.subplot(1, 2, 2)
    plt.hist(np.log1p(train['revenue']))
    plt.title('Distribution of log of revenue')
    plt.show()
    # 画预算原始图和np.log1p后的图
    fig, ax = plt.subplots(figsize=(16, 6))
    plt.subplot(1, 2, 1)
    plt.hist(train['budget'])
    plt.title('Distribution of budget')
    plt.subplot(1, 2, 2)
    plt.hist(np.log1p(train['budget']))
    plt.title('Distribution of log of budget')
    plt.show()
    # 画是否有主页的收入对比图
    train['has_homepage'] = 0
    train.loc[train['homepage'].isnull() == False, 'has_homepage'] = 1
    test['has_homepage'] = 0
    test.loc[test['homepage'].isnull() == False, 'has_homepage'] = 1
    sns.catplot(x='has_homepage', y='revenue', data=train)
    plt.title('Revenue for film with and without homepage')
    plt.show()
    # 语言的收入电影对比
    plt.figure(figsize=(16, 8))
    sns.boxplot(x='original_language', y='revenue', data=train.loc[
        train['original_language'].isin(train['original_language'].value_counts().head(10).index)])
    plt.title('Mean revenue per language')
    # 展示电影标题中的常见词
    plt.figure(figsize=(12, 12))
    text = ' '.join(train['original_title'].values)
    wordcloud = WordCloud(max_font_size=None, background_color='white', width=1200, height=1000).generate(text)
    plt.imshow(wordcloud)
    plt.title('Top words in titles')
    plt.axis("off")
    plt.show()
    # Top words
    plt.figure(figsize=(12, 12))
    text = ' '.join(train['overview'].fillna('').values)
    wordcloud = WordCloud(max_font_size=None, background_color='white', width=1200, height=1000).generate(text)
    plt.imshow(wordcloud)
    plt.title('Top words in overview')
    plt.axis("off")
    plt.show()
    # Revenue vs popularity
    plt.figure(figsize=(16, 8))
    plt.subplot(1, 2, 1)
    plt.scatter(train['popularity'], train['revenue'])
    plt.title('Revenue vs popularity')
    plt.subplot(1, 2, 2)
    plt.scatter(train['popularity'], train['log_revenue'])
    plt.title('Log Revenue vs popularity')
    plt.show()

    json_cols = ['genres', 'production_companies', 'production_countries', 'spoken_languages', 'Keywords', 'cast',
                 'crew']
    # 似乎大多数电影的长度为1.5-2小时，收入最高的电影也在这个范围内
    plt.figure(figsize=(20, 6))
    plt.subplot(1, 3, 1)
    plt.hist(train['runtime'].fillna(0) / 60, bins=40)
    plt.title('Distribution of length of film in hours')
    plt.subplot(1, 3, 2)
    plt.scatter(train['runtime'].fillna(0), train['revenue'])
    plt.title('runtime vs revenue')
    plt.subplot(1, 3, 3)
    plt.scatter(train['runtime'].fillna(0), train['popularity'])
    plt.title('runtime vs popularity')
    plt.show()

    for col in tqdm(json_cols + ['belongs_to_collection']):
        train[col] = train[col].apply(lambda x: get_dictionary(x))
        test[col] = test[col].apply(lambda x: get_dictionary(x))

    train_dict = get_json_dict(train)
    test_dict = get_json_dict(test)

    # remove cateogry with bias and low frequency
    for col in json_cols:
        remove = []
        train_id = set(list(train_dict[col].keys()))
        test_id = set(list(test_dict[col].keys()))

        remove += list(train_id - test_id) + list(test_id - train_id)
        for i in train_id.union(test_id) - set(remove):
            if train_dict[col][i] < 10 or i == '':
                remove += [i]

        for i in remove:
            if i in train_dict[col]:
                del train_dict[col][i]
            if i in test_dict[col]:
                del test_dict[col][i]

    all_data = prepare(pd.concat([train, test]).reset_index(drop=True))
    train = all_data.loc[:train.shape[0] - 1, :]
    test = all_data.loc[train.shape[0]:, :]
    print(train.columns)
    print(train.shape)
    train.to_csv('./X_train.csv', index=False)
    test.to_csv('./X_test.csv', index=False)
    y.to_csv('./y_train.csv', header=True, index=False)
