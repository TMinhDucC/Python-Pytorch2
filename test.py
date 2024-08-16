from lib import *

a=torch.tensor([[2.1, -1.3],
        [0.5,  1.2],
        [1.0,  0.3],
        [-0.7, 2.5],
        [0.2,  1.0]])
value, preds = torch.max(a, 1)
value1, preds1 = torch.max(a, 0)

print(value)
print(preds)
print("//////////")
print(value1)
print(preds1)

