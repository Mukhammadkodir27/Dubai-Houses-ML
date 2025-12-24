# 

# ! will be needed for testing or validation

y_pred_log = model_dt_best2.predict(X_train)

# convert logged prices back to original prices
y_true = np.exp(y_train)
y_pred = np.exp(y_pred_log)


# compute MAE in real units
from sklearn.metrics import mean_absolute_error

mae_real = mean_absolute_error(y_true, y_pred)
print("MAE in original price scale:", mae_real)

