## basic matlab/octave commands

whos                        % show all variables with size of current session
who                         % show al variable names only
clear                       % delete all variables

## matrix operations
A = [1 2; 3 4]              % khai báo ma trận 2x2
X'                          % matrix transpose
pinv(X)                     % pseudo inverse matrix
eye(n)                      % tạo ma trận đơn vị có kích thước nxn
zeros(n)                    % tạo ma trận 0 (tất cả các phần tử là 0) kích thước nxn
zeros(m, n)                 % tạo ma trận 0 kích thước mxn
ones(n)                     % tạo ma trận 1 (tất cả các phần tử là 1) kích thước nxn
ones(m,n)

load(file_name)             % đọc dữ liệu từ file_name

addpath(path)               % thêm đường dẫn cho Octave execution path

## code debugging technique
Insert a "keyboard" command in your script, just before the function ends.
This will cause the program to exit to the debugger, so you can inspect all your variables from the command line.
