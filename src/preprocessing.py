import pandas as pd
from sklearn.model_selection import train_test_split

RAW_PATH = "./data/raw/phishing_email.csv"
OUT_DIR = "./data/processed/"
RANDOM_STATE = 42

def main():
    df = pd.read_csv(RAW_PATH)

    X = df["text_combined"]
    y = df["label"]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.5,
        random_state=RANDOM_STATE,
        stratify=y_temp
    )

    train_df = pd.DataFrame({"text": X_train, "label": y_train})
    val_df   = pd.DataFrame({"text": X_val,   "label": y_val})
    test_df  = pd.DataFrame({"text": X_test,  "label": y_test})

    train_df.to_csv(OUT_DIR + "train.csv", index=False)
    val_df.to_csv(OUT_DIR + "val.csv", index=False)
    test_df.to_csv(OUT_DIR + "test.csv", index=False)

if __name__ == "__main__":
    main()
