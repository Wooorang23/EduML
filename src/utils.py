import joblib as jb

from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

def lin_regressor(X, y, file_path, test_size=0.2, random_state=42):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    lin_model = LinearRegression()
    lin_model.fit(X_train, y_train)
    
    y_pred = lin_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f'MSE (LR): {mse:.3f}')
    print(f'R^2 (LR): {r2:.3f}')

    jb.dump(lin_model, file_path)

def dt_regressor(X, y, file_path, test_size=0.2, random_state=42):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    dt_model = DecisionTreeRegressor(random_state=random_state)
    dt_model.fit(X_train, y_train)
    
    y_pred = dt_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f'MSE (DT): {mse:.3f}')
    print(f'R^2 (DT): {r2:.3f}')
    
    jb.dump(dt_model, file_path)

def xgb_regressor(X, y, file_path, test_size=0.2, random_state=42):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    xgb_model = XGBRegressor(random_state=random_state)
    xgb_model.fit(X_train, y_train)
    
    y_pred = xgb_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f'MSE (XGB): {mse:.3f}')
    print(f'R^2 (XGB): {r2:.3f}')
    
    jb.dump(xgb_model, file_path)