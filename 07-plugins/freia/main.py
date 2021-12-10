import torch

import FrEIA.framework as FF
import FrEIA.modules as FM

# 定义
class FixedRandomElementwiseMultiply(FM.InvertibleModule):
    def __init__(self, dims_in) -> None:
        super().__init__(dims_in)
        self.random_factor = torch.randint(1, 3, size=(1, dims_in[0][0]))
    
    def forward(self, x, rev=False, jac=True):
        x = x[0]
        if not rev:
            # 前向操作
            x = x * self.random_factor
            log_jacobian_det = self.random_factor.float().log().sum()
        else:
            # 后项操作
            x = x / self.random_factor
            log_jacobian_det = -self.random_factor.float().log().sum()

        return (x,), log_jacobian_det
    
    def output_dims(self, dims_in):
        return dims_in

class ConditionalSwap(FM.InvertibleModule):
    def __init__(self, dims_in, dims_c) -> None:
        super().__init__(dims_in=dims_in, dims_c=dims_c)
        
    def forward(self, x, c, rev=False, jac=True): # c means condition
        x1, x2 = x
        log_jacobian_det = 0

        x1_new = x1 + 0.
        x2_new = x2 + 0.

        for i in range(x1.size(0)):
            x1_new[i] = x1[i] if c[0][i] > 0 else x2[i]
            x2_new[i] = x2[i] if c[0][i] > 0 else x1[i]

        return (x1_new, x2_new), log_jacobian_det
    
    def output_dims(self, dims_in):
        return dims_in
        

batch_size = 2
in_dimension = 2

# 复杂使用
input1 = FF.InputNode(in_dimension, name='Input1')
input2 = FF.InputNode(in_dimension, name='Input2')

cond = FF.ConditionNode(1, name='ConditionNode')

mult1 = FF.Node(input1.out0, FixedRandomElementwiseMultiply, {}, name='mult1')
cond_swap = FF.Node([mult1.out0, input2.out0], ConditionalSwap, {}, conditions=cond, name='cond_swap')
mult2 = FF.Node(cond_swap.out1, FixedRandomElementwiseMultiply, {}, name='mult2')

output1 = FF.OutputNode(cond_swap.out0, name='output1')
output2 = FF.OutputNode(mult2.out0, name='output2')

inn_net = FF.GraphINN([
    input1, input2, cond,
    mult1, cond_swap, mult2,
    output1, output2
])

x1 = torch.randn(batch_size, in_dimension) * 10 // 1
x2 = torch.randn(batch_size, in_dimension) * 10 // 1
c = torch.randn(batch_size) // 1

(z1, z2), det = inn_net([x1, x2], c=c)
(x1_rev, x2_rev), _ = inn_net([z1, z2], c=c,  rev=True, jac=False)

print(x1)
print(x1_rev)
