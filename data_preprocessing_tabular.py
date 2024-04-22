import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

sc = StandardScaler()

def preprocess_compas(split, pred_assign=0):
    seed = 0
    data = pd.read_csv(
        "https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv")
    data = data[(data['days_b_screening_arrest'] <= 30) &
                (data['days_b_screening_arrest'] >= -30) &
                (data['is_recid'] != -1) &
                (data['c_charge_degree'] != "O") &
                (data['score_text'] != "N/A")]

    data = data[(data['race'] == 'African-American') | (data['race'] == 'Caucasian')]

    print("Original data columns: ", data.columns)

    data = data[["sex", "age", "age_cat", "race", "juv_fel_count", "juv_misd_count",
                 "juv_other_count", "priors_count", "c_charge_degree", "two_year_recid", "decile_score", "decile_score.1", "v_decile_score",
                 ]]
    data = data.assign(sex=(data["sex"] == "Male") * 1)
    data = pd.get_dummies(data)
    data = data.dropna()

    data['decile_score'] = data['decile_score'] * 0.2

    data = data[["sex", "age", "juv_fel_count", "juv_misd_count",
                 "priors_count", "two_year_recid",
                 "age_cat_25 - 45", "age_cat_Greater than 45", "race_Caucasian", "c_charge_degree_M",
                 "decile_score.1", "juv_other_count", "decile_score", "v_decile_score"]]

    index_train = int(len(data) * split)
    data1 = data.iloc[0:index_train].sort_values(by=['decile_score.1'], ascending=True)
    data2 = data.iloc[index_train:]
    data = data1.append(data2)

    print("data: ", data)
    print("columns: ", data.columns)

    X_total = data.iloc[:, 0:12].values

    y_total = data.iloc[:, 12].values

    X_train, X_test, y_train, y_test = train_test_split(X_total, y_total, test_size=0.3, random_state=seed,
                                                        shuffle=False)

    X_observe, X_unobserve, y_observe, y_unobserve = train_test_split(X_train, y_train, test_size=0.3,
                                                                      random_state=seed, shuffle=False)

    sc = StandardScaler()
    X_observe = sc.fit_transform(X_observe)
    X_unobserve = sc.transform(X_unobserve)
    X_train = np.r_[X_observe, X_unobserve]
    X_test = sc.transform(X_test)

    train_sel_var = np.concatenate([np.ones(y_observe.shape), np.zeros(y_unobserve.shape)])

    pred_feats = [0, 1, 3, 4, 6, 7, 9]

    if pred_assign == 0:
        dataset = dict(
            train_X_s=np.concatenate([X_train[:, 0:10]], axis=1),
            # train_X_p=X_train[:, 0:8],
            train_X_p=X_train[:, pred_feats],
            test_X_s=np.concatenate([X_test[:, 0:10]], axis=1),
            # test_X_p=X_test[:, 0:8],
            test_X_p=X_test[:, pred_feats],
            train_y=y_train,
            test_y=y_test,
            train_sel_var=train_sel_var
        )
    else:
        dataset = dict(
            train_X_s=X_train[:, 0:10],
            test_X_s=X_test[:, 0:10],
            train_y=y_train,
            test_y=y_test,
            train_sel_var=train_sel_var
        )

    return dataset



def preprocess_crime(split, pred_assign=0):
    attrib = pd.read_csv('./crime_data/attributes.csv', delim_whitespace=True)
    data = pd.read_csv('./crime_data/communities.data', names=attrib['attributes'])
    data = data.drop(columns=['state', 'county', 'community', 'communityname', 'fold'], axis=1)

    data = data.replace('?', np.nan)

    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

    imputer = imputer.fit(data[['OtherPerCap']])
    data[['OtherPerCap']] = imputer.transform(data[['OtherPerCap']])
    data = data.dropna(axis=1)

    # Sort data by NumUnderPov attribute
    index_train = int(len(data) * split)
    data1 = data.iloc[0:index_train].sort_values(by=['NumUnderPov'], ascending=True)
    data2 = data.iloc[index_train:]
    data = data1.append(data2)

    print("CRIME columns: ", data.columns[:100])
    print("CRIME label: ", data.columns[100])

    X_total = data.iloc[:, 0:100].values
    y_total = data.iloc[:, 100].values

    X_train, X_test, y_train, y_test = train_test_split(X_total, y_total, train_size=split,
                                                        shuffle=False)

    X_observe, X_unobserve, y_observe, y_unobserve = train_test_split(X_train, y_train, train_size=split,
                                                                      shuffle=False)

    sc = StandardScaler()
    X_observe = sc.fit_transform(X_observe)
    X_unobserve = sc.transform(X_unobserve)
    X_train = np.r_[X_observe, X_unobserve]
    X_test = sc.transform(X_test)

    train_sel_var = np.concatenate([np.ones(y_observe.shape), np.zeros(y_unobserve.shape)])

    # pred_feats = [2, 4, 5, 9, 10, 11, 14, 15, 16, 18, 21, 22, 24, 25]

    pred_feats = [1, 2, 3, 5, 8, 9, 11, 15, 16, 20, 21, 24, 25, 26]

    if pred_assign == 0:
        dataset = dict(
            train_X_s=np.concatenate([X_train[:, 1:27]], axis=1),
            # train_X_p=X_train[:, 5:20],
            train_X_p=X_train[:, pred_feats],
            test_X_s=np.concatenate([X_test[:, 1:27]], axis=1),
            # test_X_p=X_test[:, 5:20],
            test_X_p=X_test[:, pred_feats],
            train_y=y_train,
            test_y=y_test,
            train_sel_var=train_sel_var
        )
    else:
        dataset = dict(
            train_X_s=X_train[:, 1:27],
            test_X_s=X_test[:, 1:27],
            train_y=y_train,
            test_y=y_test,
            train_sel_var=train_sel_var
        )

    return dataset
