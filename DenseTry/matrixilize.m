q = cell(1,size(fin_output,1));
for j = 1:size(fin_output)
    q{j} = reshape(fin_output(j,:),[28 28])';
end