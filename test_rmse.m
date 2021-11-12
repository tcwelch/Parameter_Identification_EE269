% y and y_pred have [zeta, wn] as rows
function [overall_prmse, prmse_zeta,prmse_wn] = test_rmse(y,y_pred)
    overall_prmse = rms(reshape(y,[],1) - reshape(y_pred,[],1));
    prmse_zeta = rms(y(:,1) - y_pred(:,1))*(100*N/sum(y(:,1)));
    prmse_wn = rms(y(:,2) - y_pred(:,2))*(100*N/sum(y(:,2)));
end
