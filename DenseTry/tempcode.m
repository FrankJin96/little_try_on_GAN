tx = cell(1,10000);
for j = 1:10000
    tx{j} = train_y(j,:);
    tx{j} = reshape(tx{j},28,28);
    tx{j} = tx{j}';
end

for i = 1:400
    visit(i,tx);
end