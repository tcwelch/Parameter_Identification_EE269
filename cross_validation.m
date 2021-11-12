%This function can be called on y and y_pred for [zeta, wn] as rows or with
%just zeta or just wn labels and predictions.
function [overall_prmse] = cross_validation(y1,y2,y3,y4,y5,y_pred1,y_pred2,y_pred3,y_pred4,y_pred5)
    N = length(reshape(y_pred1,[],1));
    prmse1 = rms(reshape(y1,[],1) - reshape(y_pred1,[],1))*(100*N/sum(reshape(y1,[],1)));
    prmse2 = rms(reshape(y2,[],1) - reshape(y_pred2,[],1))*(100*N/sum(reshape(y2,[],1)));
    prmse3 = rms(reshape(y3,[],1) - reshape(y_pred3,[],1))*(100*N/sum(reshape(y3,[],1)));
    prmse4 = rms(reshape(y4,[],1) - reshape(y_pred4,[],1))*(100*N/sum(reshape(y4,[],1)));
    prmse5 = rms(reshape(y5,[],1) - reshape(y_pred5,[],1))*(100*N/sum(reshape(y5,[],1)));
    overall_prmse = mean([prmse1,prmse2,prmse3,prmse4,prmse5]);
end