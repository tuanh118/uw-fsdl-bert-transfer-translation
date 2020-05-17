from PipelineModel import PipelineModel

# Load data
with open('x_test.fr-en.fr', 'r', encoding='utf-8') as xfile:
    x = xfile.read().splitlines()

with open('y_test_true.fr-en.fr', 'r', encoding='utf-8') as yfile:
    y = yfile.read().splitlines()

print(x[0])
print(y[0])

# model = PipelineModel()
