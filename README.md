# [Machine Learning cơ bản](https://machinelearningcoban.com/about/)
## [Softmax Regression](https://machinelearningcoban.com/2017/02/17/softmax/)
### 1. Bài toán
Xét ánh xạ  

$$\begin{aligned} g: \mathbb{R}^m &\to \begin{Bmatrix} 0,1 \end{Bmatrix}^C,C>2 \\\\ (x_1,x_2,\dots,x_m) &\mapsto g(x_1,x_2,\dots,x_m) \end{aligned}$$

Giả sử có n điểm dữ liệu trong không gian m-chiều với mỗi  $i = 1,\dots,n$ thì $g(x_1^{(i)},x_2^{(i)},\dots,x_m^{(i)}) = \mathbf{y_i}$, chúng ta cần tìm một hàm số $f$ sao cho 
$$f(x_1,x_2,\dots,x_m)=\theta(\mathbf{W^T x}) = \left(\theta(\mathbf{w ^T_1 x}),\theta(\mathbf{w^T_2 x}),\dots,\theta(\mathbf{w^T_C x})\right) \approx g(x_1,x_2,\dots,x_m)$$    
Thay vì 'định lượng' ta sử dụng 'định tính' để chọn hàm $\theta$.
  * Các $\mathbf{w^T_i x}$ phải dương và tổng của chúng bằng 1 . 
  * Giá trị $\mathbf{w^T_i x}$ càng lớn thì xác suất dữ liệu rơi vào lớp $i$ càng cao, do đó ta cần một hàm đồng biến.
  * $\mathbf{w^T_i x}$ có thể nhận giá trị cả âm và dương. Do đó ta cần một hàm số mượt biến $\mathbf{w^T_i x}$ thành một giá trị dương và đồng biến.
   
Vậy hàm $\theta$ cần tìm là:
$$\theta(\mathbf{w^T_i x})=e^{\mathbf{w^T_i x}}\div \sum_{j=1}^{C} e^{\mathbf{w^T_j x}}, ~~ \forall i = 1, 2, \dots, C$$
Lúc này ta có thể giả sử rằng:
$$P(\mathbf{y_i}|\mathbf{x_i};\mathbf{W})=\sum_{j=1}^C a_{ji}^{y_{ji}},~~a_{ji} = \theta(\mathbf{w^T_j x_i})$$
Xét toàn bộ dữ liệu với $\mathbf{X=(x_1,x_2,\dots,x_n)}$ và $\mathbf{Y=(y_1,y_2,\dots,y_n)},$ chúng ta cần tìm $\mathbf{W}$ để cho biểu thức sau đạt giá trị lớn nhất:  
$$P(\mathbf{Y}|\mathbf{X}; \mathbf{W})$$
Giả sử rằng các điểm dữ liệu là ngẫu nhiên độc và lập với nhau, ta có thể viết:

$$P(\mathbf{Y}|\mathbf{X}; \mathbf{W}) =\prod_{i=1}^n P(\mathbf{Y}|\mathbf{X}; \mathbf{W}) = \prod_{i=1}^n \prod_{j=1}^C a_{ji}^{y_{ji}}$$  

Trực tiếp tối ưu hàm số này theo $\mathbf{W}$ không đơn giản, do đó ta sẽ tối ưu hàm số sau:
$$\mathbf{J(W;X,Y)} = -\log P(\mathbf{Y}|\mathbf{X}; \mathbf{W}) =- \sum_{i=1}^n \sum_{j=1}^C y_{ji} \log(a_{ji})$$

Hàm mất mát với chỉ một điểm dữ liệu $(\mathbf{x_i},\mathbf{y_i})$ là:  

$$\mathbf{J(W;x_i,y_i)} =- \sum_{j=1}^C y_{ji} \log(a_{ji}) = -\sum_{j=1}^C y_{ji} \mathbf{w}_j^T\mathbf{x}_i + log \left( \sum \right)$$

