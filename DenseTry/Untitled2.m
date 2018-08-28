sum = 0;
for i = 1:10000
    for j = 1:784
        if test_x(i,j) > 0
            sum = sum + 1;
        end
    end
end
