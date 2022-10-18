function HVC = CalHVC(data,ref)
    [~,N,M]  = size(data);
    ref = ones(1,M)*ref;
    data = reshape(data,N,M);
    HVC = zeros(1,N);

    for i=1:N
        data1 = data;
        s = data1(i,:);
        data1(i,:)=[];
        data1 = max(s,data1);        
        HVC(1,i) = prod(ref-s)-stk_dominatedhv(data1,ref); 
    end
end