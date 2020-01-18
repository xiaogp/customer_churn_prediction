import pickle

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_auc_score


continus_cols = ["shop_duration", "recent", "monetary", "max_amount", "items_count", 
                 "valid_points_sum", "member_day", "frequence", "avg_amount", "item_count_turn", 
                 "avg_piece_amount", "monetary3","max_amount3", "items_count3", 
                 "frequence3", "shops_count", "promote_percent", "wxapp_diff", "store_diff", 
                 "week_percent"]

category_cols = ["CHANNEL_NUM_ID", "shop_channel", "infant_group", "water_product_group", 
                 "meat_group", "beauty_group", "health_group", "fruits_group", "vegetables_group", 
                 "pets_group", "snacks_group", "smoke_group", "milk_group", "instant_group", 
                 "grain_group"]


if __name__ == "__main__":
    df = pd.read_csv("./data/churn.csv")
    train, test = train_test_split(df, test_size=0.2)
    
    # 整体组装
    preprocessor = ColumnTransformer(
            transformers=[
                    ("0_fillna", SimpleImputer(strategy='constant', fill_value=0), continus_cols), 
                    ("onehot", OneHotEncoder(handle_unknown='ignore'), category_cols)
                    ])
    clf = Pipeline(steps=[("preprocessor", preprocessor), 
                          ("classifier", XGBClassifier(max_depth=8, n_estimators=100, n_jods=3))])
    
    # 训练
    clf.fit(train[continus_cols + category_cols], train["label"])
    pickle.dump(clf, open("./churn_xgb.model", "wb"))
    
    # 预测
    model = pickle.load(open("./churn_xgb.model", "rb"))
    predictions = model.predict(test[continus_cols + category_cols])
    predict_proba = model.predict_proba(test[continus_cols + category_cols])[:, 1]
    
    # 模型评价
    print("acc:", accuracy_score(test["label"], predictions))
    print("pri:", precision_score(test["label"], predictions))
    print("rec:", recall_score(test["label"], predictions))
    print("auc:", roc_auc_score(test["label"], predict_proba))
