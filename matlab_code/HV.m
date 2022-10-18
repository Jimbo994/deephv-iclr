function Score = HV(PopObj,r)
% <metric> <max>
% Hypervolume

%------------------------------- Reference --------------------------------
% E. Zitzler and L. Thiele, Multiobjective evolutionary algorithms: A
% comparative case study and the strength Pareto approach, IEEE
% Transactions on Evolutionary Computation, 1999, 3(4): 257-271.
%------------------------------- Copyright --------------------------------
% Copyright (c) 2018-2019 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    % Normalize the population according to the reference point set
    [~,N,M]  = size(PopObj);
    PopObj = reshape(PopObj,N,M);
    %PopObj(any(PopObj>r,2),:) = [];
    % The reference point is set to (1,1,...)
    RefPoint = r*ones(1,M);
    %RefPoint = zeros(1,M)

    if isempty(PopObj)
        Score = 0;
    else
        Score = HVExact(PopObj,RefPoint);
    end
end
