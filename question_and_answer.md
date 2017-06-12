
## Một số câu hỏi

công thức normal equation:  
theta = pinv(X' * X) * X' * y  

Q1: Normal equation cho nghiệm theta tại local minimum hay global minimum của J function?  
A1:  

Q2: Tìm lambda trước hay tìm theta trước?  
Theo như video "Regularized Linear Regression" thì sau khi thêm hệ số lambda vào công thức rồi mới tìm theta.  
A2: Tìm theta trước  

Q3: Chọn giá trị lambda (regulation parameter) như thế nào? (tiêu chuẩn, phương pháp chọn)  
A3: Giống với chọn alpha, thử các giá trị khác nhau  

Q4: Giải thích công thức đạo hàm J function của logistic regression sang gradient descent?  
A4: (e^x)' = e^x  
Đạo hàm của hàm e^x bằng chính nó.  

Q5: Trong neural network thì số lượng phần tử của hidden layer thứ j (Sj) là do mình tùy chọn đúng ko?  
A5:  

Q6: Tại sao neural network lại dùng logistic function (sigmoid/activation function) mà không dùng linear regresstion function?   
Bài toán regresstion và bài toán classification nếu sử dụng neural network thì đều dùng logistic function cho từng node đúng ko?   
A6:  

Q7: Bao nhiêu layer là đủ đối với 1 neural network? Hay nói cách khác làm sao thế xác định được số lượng layer cần thiết cho một mô hình?  
A7:  

Q8: Mỗi lần học thì neural network sẽ thay đổi các hệ số để phù hợp với các input khác nhau đúng ko?  
A8:  

Q9: Trong regression và classification sử dụng cost function để đánh giá quá trình học. Trong neural network thì tiêu chuẩn đánh giá quá trình trình học của từng hidden node là gì?  
Hay khi nào thì dừng quá trình tính hệ số theta cho từng node trong network?  
A9:  

