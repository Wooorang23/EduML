import pandas as pd
from config import DATA_PATH
from utils import lin_regressor, dt_regressor, xgb_regressor

# preprocessing and feature engineering
data = pd.read_csv(DATA_PATH + 'EduML/data/external/student_performance.csv')
file_path = 'models/gpa_target/'

features = data.columns.drop(['StudentID', 'GPA'])
X = data[features]
X = pd.get_dummies(X, 
                   columns=['Ethnicity', 'ParentalEducation', 'GradeClass', 'Music', 'Extracurricular', 'Sports'], 
                   drop_first=True)
y = data.GPA

# model training
lr = lin_regressor(X, y, file_path + 'EduML-lin.pkl')
dt = dt_regressor(X, y, file_path + 'EduML-dt.pkl')
xgb = xgb_regressor(X, y, file_path + 'EduML-xgb.pkl')