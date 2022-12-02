import os
import json

database_filename = os.path.join(os.path.dirname(__file__), "src", "projeto-1", "dimensions_projeto_1.json")

li, ls, step = 5, 50, 1

dimensions = []

for i in range(li, ls + 1):
    for j in range(li, ls + 1):
        for k in range(li, ls + 1):
            dimensions.append([i / 100, j / 100, (j * 4) / 100, k / 100])

dimensions_json = json.dumps({
    "dimensions": dimensions
})

try:
    file = open(database_filename, "x")
    file.write(dimensions_json)
    file.close()
    print("Os dados foram criados")
except:
    print("O arquivo de dados já existe")