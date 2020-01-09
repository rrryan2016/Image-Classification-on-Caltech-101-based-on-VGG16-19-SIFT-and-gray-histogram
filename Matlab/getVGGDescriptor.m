function hist = getVGGDescriptor(net,im)
    insz = net.Layers(1).InputSize;
    if size (im,3) ==1
        rgb = cat(3,im,im,im);
        im = mat2gray(rgb);
    end
    img = single(im);
    img_resize = imresize(img,insz(1:2));
    hist = activations(net,img_resize,'fc7','OutputAs','rows' ); % add transposition
    hist = hist';
end 