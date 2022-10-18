function Data = generateTrainingData(M,Data,data_num,num_on_triPF, num_on_invtriPF, num_on_random,seed)
% 这个函数是在PF上采样来构造solution set，
% -----------task-------------------
% dataset_num 100,000 -> 1,000,000
% dataset:
%   正三角PF上: 50,000
%   倒三角PF上: 50,000
%   random generated: 900,000
% -----------task-------------------
    a = 1.5;
    b = 0.5; % a和b两个参数是控制下面的p的取值范围
    rng(seed);    
    tic
    % generate solutions for triangular and inverted triangular PF. 
    for i=1:num_on_triPF
        if mod(i, 10000) == 0
            disp(['i=',num2str(i),'/',num2str(num_on_triPF+num_on_invtriPF+num_on_random)]);
            toc
        end
        num = 0;
        while num == 0
            num = ceil(data_num*rand);
        end
        p = rand*a+b; % p是用来控制PF的曲率
        temp = abs(UniformSphere_ExponentioalPowerDistribution(num,ones(1,M)*p,1));
        Data(i,1:num,:) = temp';
    end
    for i=num_on_triPF+1:num_on_triPF+num_on_invtriPF
        if mod(i, 10000) == 0
            disp(['i=',num2str(i),'/',num2str(num_on_triPF+num_on_invtriPF+num_on_random)]);
            toc
        end
        num = 0;
        while num == 0
            num = ceil(data_num*rand);
        end
        p = rand*a+b; % p是用来控制PF的曲率
        temp = abs(UniformSphere_ExponentioalPowerDistribution(num,ones(1,M)*p,1));
        temp = temp*(-1)+1;
        Data(i,1:num,:) = temp';
        %HVC(:,i) = CalHVC(Data(:,:,i),r,data_num);  
    end

    % generate random non-dominated solutions. 
    for i=num_on_triPF+num_on_invtriPF+1:num_on_triPF+num_on_invtriPF+num_on_random
        if mod(i, 10000) == 0
            disp(['i=',num2str(i),'/',num2str(num_on_triPF+num_on_invtriPF+num_on_random)]);
            toc
        end
        
        % the dataset contain num of points
        num = 0;
        while num == 0
            num = ceil(data_num*rand);
        end
        % make sure the solutions contain at least 1 Front containing more
        % than num points.
        check_num_points = 0;      % largest num of points for Fronts
        while check_num_points < num
            % random generate 100*data_num points in [0,1]^M
            solutions = rand(10*data_num,M);
            % get non-dominated solutions
            [FrontNo,MaxFNo] = NDSort(solutions,10*data_num);
            % check exist Front whose num_points exceed num
            for front_idx = 1:MaxFNo
                num_points = sum(FrontNo==front_idx);
                if check_num_points < num_points
                    check_num_points = num_points;
                end
            end
        end
        
        % make sure the selected Front containing more than num points
        check_num_points = 0;
        while check_num_points < num
            selectedNo = ceil(MaxFNo*rand); 
            ndsolutions = solutions(FrontNo==selectedNo,:); 
            if check_num_points < size(ndsolutions, 1)
                check_num_points = size(ndsolutions, 1);
            end
        end
            
        % check num of ndsolutions > num, than randomly select num of
        % points. 
        if size(ndsolutions,1) > num
            selected_ndsolutions = ndsolutions(randperm(size(ndsolutions,1),num),:);
            Data(i,1:num,:) = selected_ndsolutions;
        else    % exactly num points. 
            Data(i,1:size(ndsolutions,1),:) = ndsolutions;
        end
    end
end