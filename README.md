# [Machine Learning cơ bản](https://machinelearningcoban.com/about/)
## [Softmax Regression](https://machinelearningcoban.com/2017/02/17/softmax/)
### 1. Bài toán
Xét ánh xạ  

$$\begin{aligned} g: \mathbb{R}^m &\to \begin{Bmatrix} 0,1 \end{Bmatrix}^C,C>2 \\\\ (x_1,x_2,\dots,x_m) &\mapsto g(x_1,x_2,\dots,x_m) \end{aligned}$$

Giả sử có n điểm dữ liệu trong không gian m-chiều với mỗi  $i = 1,\dots,n$ thì $g(x_1^{(i)},x_2^{(i)},\dots,x_m^{(i)}) = \mathbf{y}_i$, chúng ta cần tìm một hàm số $f$ sao cho 
$$f(x_1,x_2,\dots,x_m)=\theta(\mathbf{W}^T \mathbf{x}) = \left(\theta(\mathbf{w}_1^T \mathbf{x}),\theta(\mathbf{w}_2^T \mathbf{x}),\dots,\theta(\mathbf{w}_C^T \mathbf{x})\right) \approx g(x_1,x_2,\dots,x_m)$$    
Thay vì 'định lượng' ta sử dụng 'định tính' để chọn hàm $\theta$.
  * Các $\mathbf{w}_i^T \mathbf{x}$ phải dương và tổng của chúng bằng 1 . 
  * Giá trị $\mathbf{w}_i^T \mathbf{x}$ càng lớn thì xác suất dữ liệu rơi vào lớp $i$ càng cao, do đó ta cần một hàm đồng biến.
  * $\mathbf{w}_i^T \mathbf{x}$ có thể nhận giá trị cả âm và dương. Do đó ta cần một hàm số mượt biến $\mathbf{w}_i^T \mathbf{x}$ thành một giá trị dương và đồng biến.
   
Vậy hàm $\theta$ cần tìm là:
$$\theta(\mathbf{w}\_i^T \mathbf{x}) = \exp({\mathbf{w}_i^T \mathbf{x}}) \div \sum\_{j=1}^{C} \exp({\mathbf{w}_j^T \mathbf{x}}), ~~ \forall i = 1, 2, \dots, C$$  

Lúc này ta có thể giả sử rằng:

$$P(\mathbf{y}\_i | \mathbf{x}\_i;\mathbf{W}) = \sum_{j=1}^C a_{ji}^{y_{ji}},~~a_{ji} = \theta(\mathbf{w}_j^T \mathbf{x}_i)$$

Xét toàn bộ dữ liệu với $\mathbf{X}=(\mathbf{x}_1,\mathbf{x}_2,\dots,\mathbf{x}_n)$ và $\mathbf{Y}=(\mathbf{y}_1,\mathbf{y}_2,\dots,\mathbf{y}_n),$ chúng ta cần tìm $\mathbf{W}$ để cho biểu thức sau đạt giá trị lớn nhất:  
$$P(\mathbf{Y}|\mathbf{X}; \mathbf{W})$$
Giả sử rằng các điểm dữ liệu là ngẫu nhiên độc và lập với nhau, ta có thể viết:

$$P(\mathbf{Y}|\mathbf{X}; \mathbf{W}) =\prod_{i=1}^n P(\mathbf{Y}|\mathbf{X}; \mathbf{W}) = \prod_{i=1}^n \prod_{j=1}^C a_{ji}^{y_{ji}}$$  

Trực tiếp tối ưu hàm số này theo $\mathbf{W}$ không đơn giản vì khi $n$ lớn tích là một số quá nhỏ dẫn tới sai số trong tính toán, do đó ta sẽ tối ưu hàm số sau:
$$J(\mathbf{W};\mathbf{X},\mathbf{Y}) = -\log P(\mathbf{Y}|\mathbf{X}; \mathbf{W}) =- \sum_{i=1}^n \sum_{j=1}^C y_{ji} \log(a_{ji})$$

Hàm mất mát với chỉ một điểm dữ liệu $(\mathbf{x}_i,\mathbf{y}_i)$ là:  

$$J(\mathbf{W};\mathbf{x}\_i,\mathbf{y}\_i) =- \sum_{j=1}^C y_{ji} \log(a_{ji}) = -\sum_{j=1}^C y_{ji} \mathbf{w}_j^T\mathbf{x}_i + log \left( \sum\_{k=1}^C \exp(\mathbf{w}_k^T \mathbf{x_i})\right)$$

Đạo hàm của hàm trên là:
$$\frac{\partial J_i(\mathbf{W})}{\partial \mathbf{W}} = \left[\frac{\partial J_i(\mathbf{W})}{\partial \mathbf{w}_1}, \frac{\partial J_i(\mathbf{W})}{\partial \mathbf{w}_2}, \dots, \frac{\partial J_i(\mathbf{W})}{\partial \mathbf{w}_C}\right]$$
Trong đó: 
$$\frac{\partial J_i(\mathbf{W})}{\partial \mathbf{w}_j} = -y\_{ji}\mathbf{x}_i + \left(\exp(\mathbf{w}_j^T\mathbf{x}_i) \div \sum\_{k = 1}^C \exp(\mathbf{w}_k^T\mathbf{x}_i)\right) \mathbf{x}_i = -y\_{ji}\mathbf{x}_i + a\_{ji} \mathbf{x}_i = \mathbf{x}_i (a\_{ji} - y\_{ji}) = e\_{ji} \mathbf{x}_i$$
Suy ra: 
$$\frac{\partial J_i(\mathbf{W})}{\partial \mathbf{W}} = \mathbf{x}_i \[e\_{1i}, e\_{2i}, \dots, e\_{Ci}\] = \mathbf{x}_i \mathbf{e}_i^T$$
Vậy công thức cập nhật (theo thuật toán SGD) cho logistic regression là:
$$\mathbf{W} = \mathbf{W} +\eta \mathbf{x}\_{i}(\mathbf{y}_i - \mathbf{a}_i)^T$$
### 2. Tài liệu tham khảo
1. [Softmax Regression - Machine Learning cơ bản](https://machinelearningcoban.com/2017/02/17/softmax/)


