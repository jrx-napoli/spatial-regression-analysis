from spreg import OLS, ML_Lag, ML_Error, GM_Lag


def run(y, y_name, x, x_names, w, w_name, ds_name):
    ols = OLS(y, x, w, spat_diag=True, moran=True, name_y=y_name, name_x=x_names, name_w=w_name, name_ds=ds_name)
    print(ols.summary)

    ml_lag = ML_Lag(y, x, w, name_y=y_name, name_x=x_names, name_w=w_name, name_ds=ds_name)
    print(ml_lag.summary)

    ml_error = ML_Error(y, x, w, name_y=y_name, name_x=x_names, name_w=w_name, name_ds=ds_name)
    print(ml_error.summary)

    gm_lag = GM_Lag(y, x, w, name_y=y_name, name_x=x_names, name_w=w_name, name_ds=ds_name)
    print(gm_lag.summary)
