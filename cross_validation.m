%This function can be called on y and y_pred for [zeta, wn] as rows or with
%just zeta or just wn labels and predictions.
function [overall_rmse] = cross_validation(y1,y2,y3,y4,y5,y_pred1,y_pred2,y_pred3,y_pred4,y_pred5)
    rmse1 = rms(reshape(y1,[],1) - reshape(y_pred1,[],1));
    rmse2 = rms(reshape(y2,[],1) - reshape(y_pred2,[],1));
    rmse3 = rms(reshape(y3,[],1) - reshape(y_pred3,[],1));
    rmse4 = rms(reshape(y4,[],1) - reshape(y_pred4,[],1));
    rmse5 = rms(reshape(y5,[],1) - reshape(y_pred5,[],1));
    overall_rmse = mean([rmse1,rmse2,rmse3,rmse4,rmse5]);
end