from torch import nn, tensor
import numpy as np

m = nn.BatchNorm2d(num_features=3, momentum=0.1)
input_py = tensor(np.array([[[[0.1, 0.2], [0.3, 0.4]],
                          [[0.5, 0.6], [0.7, 0.8]],
                          [[0.9, 1], [1.1, 1.2]]]]).astype(np.float32))
output = m(input_py)
print(output.detach().numpy())
# [[[[-1.3411044  -0.44703478]
#    [ 0.4470349   1.3411044 ]]
#
#   [[-1.3411043  -0.44703442]
#    [ 0.44703496  1.3411049 ]]
#
#   [[-1.3411039  -0.44703427]
#    [ 0.44703534  1.341105  ]]]]

# MindSpore
from mindspore import Tensor, nn
import numpy as np

m = nn.BatchNorm2d(num_features=3, momentum=0.9)
m.set_train()
#  BatchNorm2d<num_features=3, eps=1e-05, momentum=0.9, gamma=Parameter (name=gamma, shape=(3,), dtype=Float32, requires_grad=True), beta=Parameter (name=beta, shape=(3,), dtype=Float32, requires_grad=True), moving_mean=Parameter (name=mean, shape=(3,), dtype=Float32, requires_grad=False), moving_variance=Parameter (name=variance, shape=(3,), dtype=Float32, requires_grad=False)>

input_ms = Tensor(np.array([[[[0.1, 0.2], [0.3, 0.4]],
                          [[0.5, 0.6], [0.7, 0.8]],
                          [[0.9, 1], [1.1, 1.2]]]]).astype(np.float32))
output = m(input_ms)
print(output)
# [[[[-1.3411045  -0.4470348 ]
#    [ 0.44703496  1.3411045 ]]
#
#   [[-1.341105   -0.4470351 ]
#    [ 0.44703424  1.3411041 ]]
#
#   [[-1.3411034  -0.44703388]
#    [ 0.44703573  1.3411053 ]]]]