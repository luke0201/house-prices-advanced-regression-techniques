# House Prices - Advanced Regression Techniques

This is my Kaggle submission code.

## Requirements

You need Python 3.6 or higher version.

The following libraries are required.

- numpy
- scipy
- pandas
- scikit-learn
- xgboost
- lightgbm

## Usage

Download the dataset using the following command.

```
kaggle competitions download -c house-prices-advanced-regression-techniques
unzip house-prices-advanced-regression-techniques.zip && rm bike-sharing-demand.zip
```

Then run `house-prices-advanced-regression-techniques.py` as follows.

```
python house-prices-advanced-regression-techniques.py
```

Finally, submit the result using the following command. Replace the `<submission_message>` by yours.

```
kaggle competitions submit -c house-prices-advanced-regression-techniques -f submission.csv -m <submission_message>
```
