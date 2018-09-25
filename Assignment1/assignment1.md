# problem 01
(1) for example $X_i$, $net_{ik}=\omega_{ik}X_i+b_{ij}$, $z_{ik}=\frac{e^{net_{ik}}}{\sum_{k=1}^{c}e^{net_{ik}}}$.     
the lost function: $L_i=-\sum_{k=1}^c[y_{ik}log(z_{ik})]$     
the cost function: $C=\frac{1}{N}\sum_{i=1}^NL_i=-\frac{1}{N}\sum_{i=1}^N\sum_{k=1}^c[y_{ik}log(z_{ik})$    
the gradient of hidden-to-output weights: $\frac{\partial C}{\partial net_{ik}}=\frac{\partial C}{\partial z_{ij}}\frac{\partial z_{ij}}{\partial net_{ik}}$   
in this formula, $\frac{\partial C}{\partial z_{ij}}=-\frac{1}{N}\sum_{i=1}^N\sum_{j=1}^c[\frac{y_{ij}}{z_{ij}}]$   
when $j=k$: $\frac{\partial z_{ik}}{\partial net_{ik}}=\frac{\partial}{\partial net_{ik}}\frac{e^{net_{ik}}}{\sum_{k=1}^ce^{net_{ik}}}=\frac{e^{net_{ik}}\sum_{k=1}^ce^{net_{ik}}-e^{2net_{ik}}}{(\sum_{k=1}^ce^{net_{ik}})^2}=z_{ik}-z_{ik}^2$    
when $j\neq k$: $\frac{\partial z_{ij}}{\partial net_{ik}}=\frac{\partial}{\partial net_{ik}}\frac{e^{net_{ij}}}{\sum_{k=1}^ce^{net_{ik}}}=-\frac{e^{net_{ij}}e^{net_{ik}}}{(\sum_{k=1}^ce^{net_{ik}})^2}=-z_{ij}z_{ik}$
take them into the original fomula, $\frac{\partial C}{\partial net_{ik}}=-\frac{1}{N}\sum_{i=1}^N(\sum_{j\neq k}^c\frac{y_{ij}}{z_{ij}}(-z_{ij}z_{ik})+\frac{y_{ik}}{z_{ik}}(z_{ik}-z_{ik}^2))=-\frac{1}{N}\sum_{i=1}^N(\sum_{j=1}^c(-y_{ij}z_{ik})+y_{ik})$   
(2)