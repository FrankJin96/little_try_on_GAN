function visit(k,q)
filename = strcat('res_',num2str(k),'.jpg');
imwrite(q{k},filename);
% imshow(filename)
end