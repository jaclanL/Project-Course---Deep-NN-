function [error_avg, error_joints] = normalized_mse(true,pred)
    % true (nSamples x nJoints)
    % pred (nSamples x nJoints)
    % error (1 x nJoints)

    % Normalizing wrt highest value
    m = max(abs(true),[],2);
    t = true./m;
    p = pred./m;
    
    % Calculating each error value
    e = t-p;
    
    % Mean error of each joint
    error_joints = mean(e);
    error_avg = mean(error_joints);
end
   
    