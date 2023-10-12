import torch
import torch.nn as nn

a = torch.rand([64, 3, 4, 5])
print(a)

fc = nn.Linear(5, 3)
b = fc(a)

print(b)
print(b.size())