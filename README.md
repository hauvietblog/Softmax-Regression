# [Machine Learning cơ bản](https://machinelearningcoban.com/about/)
## [Softmax Regression](https://machinelearningcoban.com/2017/02/17/softmax/)
### 1. Bài toán
Xét ánh xạ  

$$\begin{aligned} g: \mathbb{R}^m &\to \begin{Bmatrix} 0,1 \end{Bmatrix}^C,C>2 \\\\ (x_1,x_2,\dots,x_m) &\mapsto g(x_1,x_2,\dots,x_m) \end{aligned}$$

Giả sử có n điểm dữ liệu trong không gian m-chiều với mỗi  $i = 1,\dots,n$ thì $g(x_1^{(i)},x_2^{(i)},\dots,x_m^{(i)}) = \mathbf{y^{(i)}}$, chúng ta cần tìm một hàm số $f$ sao cho 
$$f(x_1,x_2,\dots,x_m)=\theta(\mathbf{W^T x}) = \left(\theta(\mathbf{w ^T_1 x}),\theta(\mathbf{w^T_2 x}),\dots,\theta(\mathbf{w^T_C x})\right) \approx g(x_1,x_2,\dots,x_m)$$  
Thay vì 'định lượng' ta sử dụng 'định tính' để chọn hà m $\theta$.
  * Các $w^T_i x$ phải dương và tổng của chúng bằng 1. 
  * Giá trị $w^T_i x$ càng lớn thì xác suất dữ liệu rơi vào lớp $i$ càng cao, do đó ta cần một hàm đồng biến.
  * $w^T_i x$ có thể nhận giá trị cả âm và dương. Do đó ta cần một hàm số mượt biến $w^T_i x$ thành một giá trị dương và đồng biến.
  
Vậy hàm $\theta$ cần tìm là:
$$\theta(w^T_i x)=e^{w^T_i x}\div \sum_{j=1}^{C} e^{w^T_j x}, ~~ \forall i = 1, 2, \dots, C$$
