import pandas as pd
import numpy as np
import optuna

from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score

from xgboost import XGBClassifier


train_data_path = "/Users/sampatton/Downloads/newdata3.csv"
test_data_path = "/Users/sampatton/Downloads/teststar.csv"

train_df = pd.read_csv(train_data_path)
test_df = pd.read_csv(test_data_path)


# -------------------------
# Feature Engineering
# -------------------------
def add_features(df):
    df = df.copy()

    df["u_g"] = df["u"] - df["g"]
    df["g_r"] = df["g"] - df["r"]
    df["r_i"] = df["r"] - df["i"]
    df["i_z"] = df["i"] - df["z"]
    df["u_r"] = df["u"] - df["r"]
    df["g_i"] = df["g"] - df["i"]

    alpha_rad = np.radians(df["alpha"])
    delta_rad = np.radians(df["delta"])

    df["sin_alpha"] = np.sin(alpha_rad)
    df["cos_alpha"] = np.cos(alpha_rad)
    df["sin_delta"] = np.sin(delta_rad)
    df["cos_delta"] = np.cos(delta_rad)

    df["x_coord"] = df["redshift"] * np.cos(delta_rad) * np.cos(alpha_rad)
    df["y_coord"] = df["redshift"] * np.cos(delta_rad) * np.sin(alpha_rad)
    df["z_coord"] = df["redshift"] * np.sin(delta_rad)

    return df


train_df = add_features(train_df)
test_df = add_features(test_df)


X = train_df.drop(columns=['class'])
y_raw = train_df["class"]

X_test = test_df.drop(columns=["id"])
test_ids = test_df["id"]


# Encode classes in case they are strings
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_raw)

num_classes = len(label_encoder.classes_)

print("Classes:", label_encoder.classes_)
print("Number of classes:", num_classes)
print("Train shape after feature engineering:", X.shape)
print("Test shape after feature engineering:", X_test.shape)


preprocess = make_column_transformer(
    (
        SimpleImputer(strategy="median"),
        make_column_selector(dtype_include="number")
    ),
    (
        make_pipeline(
            SimpleImputer(strategy="most_frequent"),
            OneHotEncoder(handle_unknown="ignore")
        ),
        make_column_selector(dtype_include="object")
    )
)


cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


def objective(trial):

    model = make_pipeline(
        preprocess,
        XGBClassifier(
            objective="multi:softprob",
            num_class=num_classes,
            eval_metric="mlogloss",
            random_state=42,
            tree_method="hist",
            n_jobs=-1,

            n_estimators=trial.suggest_int("n_estimators", 200, 1800),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            max_depth=trial.suggest_int("max_depth", 2, 8),
            min_child_weight=trial.suggest_float("min_child_weight", 1, 20),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
            gamma=trial.suggest_float("gamma", 0, 5),
            reg_alpha=trial.suggest_float("reg_alpha", 0, 5),
            reg_lambda=trial.suggest_float("reg_lambda", 1, 20),
        )
    )

    score = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring="balanced_accuracy",
        n_jobs=1
    ).mean()

    return score


study = optuna.create_study(direction="maximize")

study.optimize(
    objective,
    show_progress_bar=True,
    n_trials=100
)

print("Best Parameters:")
print(study.best_params)

print("Best CV Balanced Accuracy:")
print(study.best_value)


best_model = make_pipeline(
    preprocess,
    XGBClassifier(
        objective="multi:softprob",
        num_class=num_classes,
        eval_metric="mlogloss",
        random_state=42,
        tree_method="hist",
        n_jobs=-1,
        **study.best_params
    )
)


best_model.fit(X, y)


test_preds_encoded = best_model.predict(X_test)

# Convert back to original class labels
test_preds = label_encoder.inverse_transform(test_preds_encoded)


submission = pd.DataFrame({
    "id": test_ids,
    "class": test_preds
})

submission.to_csv(
    "/Users/sampatton/Downloads/newdata.csv",
    index=False
)

print("Submission saved.")