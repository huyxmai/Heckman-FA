import statsmodels.api as sm
from sklearn.metrics import mean_squared_error

def linear_regression(data_dic, assignment):

    train_y = data_dic["train_y"].astype('float')
    test_y = data_dic["test_y"]
    train_sel_var = data_dic["train_sel_var"]

    train_X_s = data_dic["train_X_s"]
    test_X_s = data_dic["test_X_s"]

    train_X_p = train_X_s[:, assignment]
    test_X_p = test_X_s[:, assignment]

    # Add intercept
    train_X_p = sm.add_constant(train_X_p)
    test_X_p = sm.add_constant(test_X_p)

    # Train linear regression model on fully observed D_tr
    LR = sm.OLS(train_y, train_X_p)
    LR_function = LR.fit()
    beta = LR_function.params

    # Get predictions
    train_y_pred = train_X_p.dot(beta)
    train_MSE = mean_squared_error(train_y, train_y_pred)

    test_y_pred = test_X_p.dot(beta)
    test_MSE = mean_squared_error(test_y, test_y_pred)

    return train_MSE, test_MSE

def naive_linear_regression(data_dic, assignment):

    train_sel_var = data_dic["train_sel_var"]
    train_y = data_dic["train_y"].astype('float')
    test_y = data_dic["test_y"]

    train_X_s = data_dic["train_X_s"]
    test_X_s = data_dic["test_X_s"]

    # Get biased training set D_s
    train_X_p = train_X_s[:, assignment]
    train_X_p = train_X_p[train_sel_var == 1]
    train_y = train_y[train_sel_var == 1]

    test_X_p = test_X_s[:, assignment]

    # Add intercept
    train_X_p = sm.add_constant(train_X_p)
    test_X_p = sm.add_constant(test_X_p)

    # Train linear regression model on fully observed D_tr
    LR = sm.OLS(train_y, train_X_p)
    LR_function = LR.fit()
    beta = LR_function.params

    # Get predictions
    train_y_pred = train_X_p.dot(beta)
    train_MSE = mean_squared_error(train_y, train_y_pred)

    test_y_pred = test_X_p.dot(beta)
    test_MSE = mean_squared_error(test_y, test_y_pred)

    return train_MSE, test_MSE








