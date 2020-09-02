function [ dist ] = euclidean_dist( X_gallery, X_probe, model_para )
    % disp(size(X_probe))
    dist=pdist2(X_probe,X_gallery,'euclidean'); 
end

