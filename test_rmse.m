% y and y_pred have [zeta, wn] as rows
function [overall_rmse, rmse_zeta,rmse_wn] = test_rmse(y,y_pred)
    overall_rmse = rms(reshape(y,[],1) - reshape(y_pred,[],1));
    rmse_zeta = rms(y(:,1) - y_pred(:,1));
    rmse_wn = rms(y(:,2) - y_pred(:,2));
end