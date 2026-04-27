from autogluon.tabular import TabularDataset, TabularPredictor

#Data
data_url = 'https://raw.githubusercontent.com/mli/ag-docs/main/knot_theory/'
train_data = TabularDataset(f'{data_url}train.csv')
print(train_data.head())

label = 'signature'
print(train_data[label].describe())

#Training
predictor = TabularPredictor(label=label).fit(train_data)

#Prediction
test_data = TabularDataset(f'{data_url}test.csv')

y_pred = predictor.predict(test_data.drop(columns=[label]))
print(y_pred.head())

#Evaluation
print(predictor.evaluate(test_data))
print(predictor.leaderboard(test_data))
