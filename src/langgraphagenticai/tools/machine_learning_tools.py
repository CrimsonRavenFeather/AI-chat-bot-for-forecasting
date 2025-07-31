from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
import numpy as np
import pandas as pd

class MachineLearningTools:
    """
    Tools to perform machine learning task on the uploaded dataset
    """

    def __init__(self,dataset:pd.DataFrame):
        self.dataset=dataset

    def valid_datetime_series(self):
        df=self.dataset
        result = {
            "is_valid": False,
            "datetime_column": None,
            "target_column": [],
            "id_column": [],
            "Catagorical_column": [],
            "reason": ""
        }

        datetime_cols = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col]) or "date" in col.lower()]

        if not datetime_cols:
            result["reason"] = "No datetime column found."
            return result
        
        result["datetime_column"] = datetime_cols[0]
        try:
            df[result["datetime_column"]] = pd.to_datetime(df[result["datetime_column"]])
        except:
            result["reason"] = "Datetime conversion failed."
            return result

        num_cols = df.select_dtypes(include='number').columns.tolist()
        cat_cols = df.select_dtypes(include='object').columns.tolist()
        result["target_column"] = num_cols
        result["Catagorical_column"] = cat_cols
        if not num_cols:
            result["reason"] = "No numeric target column found."
            return result, df

        result["is_valid"] = True
        return result
    
    def detect_structure(self):
        result=self.valid_datetime_series()
        df=self.dataset
        if result["is_valid"]==False:
            return "Invalid Time Series"
    
        datetime_cols = result["datetime_column"]
        id_cols = result["id_column"]
        num_cols = result["target_column"]
        cat_cols = result["Catagorical_column"]

        if not datetime_cols:
            return "Not a time series dataset (no datetime column)"

        if len(cat_cols) == 0:
            if len(num_cols) == 1:
                return "Univariate Time Series"
            elif len(id_cols) == 0:
                return "Multivariate Time Series"
            elif len(id_cols) == 1:
                return "Panel Time Series"
            elif len(id_cols) > 1:
                return "Grouped Time Series"

        elif len(cat_cols) == 1 and cat_cols[0] in id_cols:
            return "Panel Time Series"

        elif len(cat_cols) > 1:
            is_hierarchical = True
            for i in range(len(cat_cols) - 1):
                parent = cat_cols[i]
                child = cat_cols[i + 1]
                mapping = df[[parent, child]].drop_duplicates()
                parent_count = mapping.groupby(child)[parent].nunique()
                if any(parent_count > 1):
                    is_hierarchical = False
                    break
            return "Hierarchical Time Series" if is_hierarchical else "Grouped Time Series"

        else:
            return "Multivariate Time Series"
    
    # convert all the possible datetime column into datetime type
    def convert_datetime(self,df):
        for col in df.columns:
            try:
                parsed_col = pd.to_datetime(df[col])
                df[col] = parsed_col
                break
            except (ValueError, TypeError):
                continue
        return df

    # do basic preprocessing task
    def preprocessing(self,df):
        df.dropna(inplace=True)
        df.drop_duplicates(inplace=True)
        return df

    # returns the target values from the dataset
    def get_target(self,df):
        return df.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # return the datetime columns from the dataset
    def get_date(self,df):
        return df.select_dtypes(include=['datetime64[ns]']).columns.tolist()
    
    # run basic ml models
    def run_basic_model(self):
        df=self.dataset
        df=self.preprocessing(df)
        df=self.convert_datetime(df)
        target_col=self.get_target(df)[0]
        datetime_col=self.get_date(df)[0]

        df = df.sort_values(by=datetime_col)
        X = df.drop(columns=[target_col, datetime_col], errors="ignore").select_dtypes(include='number')
        y = df[target_col]

        if X.empty:
            return {}
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            models = {
                "Linear Regression": LinearRegression(),
                "Random Forest": RandomForestRegressor(random_state=42),
                "Ridge": Ridge(),
                "Lasso": Lasso(),
                "Decision Tree": DecisionTreeRegressor()
            }

            results = {}
            predictions = {}

            for name, model in models.items():
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                results[name] = rmse
                predictions[name] = y_pred

            # best_model=min(results,key=results.get)
            return results