% -------------------------------------------------------------------------
function hist = getImageDescriptor(model, im)
% -------------------------------------------------------------------------
    im = standarizeImage(im) ;
    width = size(im,2) ;
    height = size(im,1) ;
    numWords = size(model.vocab, 2) ;

    % get PHOW features
    [frames, descrs] = vl_phow(im, model.phowOpts{:}) ;
    %[frames, descrs] = vl_sift(im, model.phowOpts{:}) ;

    % quantize local descriptors into visual words
    switch model.quantizer
      case 'vq'
        [drop, binsa] = min(vl_alldist(model.vocab, single(descrs)), [], 1) ;
      case 'kdtree'
        binsa = double(vl_kdtreequery(model.kdtree, model.vocab, ...
                                      single(descrs), ...
                                      'MaxComparisons', 50)) ;
    end

    for i = 1:length(model.numSpatialX)
      binsx = vl_binsearch(linspace(1,width,model.numSpatialX(i)+1), frames(1,:)) ;
      binsy = vl_binsearch(linspace(1,height,model.numSpatialY(i)+1), frames(2,:)) ;

      % combined quantization
      bins = sub2ind([model.numSpatialY(i), model.numSpatialX(i), numWords], ...
                     binsy,binsx,binsa) ;
      hist = zeros(model.numSpatialY(i) * model.numSpatialX(i) * numWords, 1) ;
      hist = vl_binsum(hist, ones(size(bins)), bins) ;
      hists{i} = single(hist / sum(hist)) ;
    end
    hist = cat(1,hists{:}) ;
    hist = hist / sum(hist) ;
end
