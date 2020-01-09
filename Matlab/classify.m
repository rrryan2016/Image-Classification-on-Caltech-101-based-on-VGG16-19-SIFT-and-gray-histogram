
% -------------------------------------------------------------------------
function [className, score] = classify(model, im)
% -------------------------------------------------------------------------
    hist = getImageDescriptor(model, im) ;
    psix = vl_homkermap(hist, 1, 'kchi2', 'gamma', .5) ;
    scores = model.w' * psix + model.b' ;
    [score, best] = max(scores) ;
    className = model.classes{best} ;
end