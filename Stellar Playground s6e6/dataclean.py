import pandas as pd
import numpy as np

train_data_path = "/Users/sampatton/Downloads/trainstar.csv"
test_data_path = "/Users/sampatton/Downloads/teststar.csv"
stellar_data = "/Users/sampatton/Downloads/star_classification.csv"

train_df = pd.read_csv(train_data_path)
test_df = pd.read_csv(test_data_path)
stellar_data = pd.read_csv(stellar_data)


# Remove bad rows
train_df = train_df[train_df["u"] > 5].reset_index(drop=True)
stellar_data = stellar_data[stellar_data["u"] > 5].reset_index(drop=True)


stellar_drop = stellar_data.drop(
    columns=[
        'obj_ID','run_ID','rerun_ID','cam_col',
        'field_ID','spec_obj_ID',
        'plate','MJD','fiber_ID'
    ]
)

train_df = train_df.drop(columns=['id'])


combined_df = pd.concat(
    [train_df, stellar_drop],
    axis=0,
    ignore_index=True
)

print(combined_df.shape)
print(combined_df.head())

combined_df.to_csv(
    "/Users/sampatton/Downloads/newdata3.csv",
    index=False
)