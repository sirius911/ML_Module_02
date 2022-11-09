import yaml
import numpy as np

# names_yaml = """
# - 'eric'
# - 'justin'
# - 'mary-kate'
# - 'cyrille'
# """
# names_yaml2 = """
# - 'Margaux'
# - 'Maxime'
# - 'Marc'
# """
# y=[1,1,1]
# names = yaml.safe_load(names_yaml)
# names2 = yaml.safe_load(names_yaml2)
# y2 = yaml.safe_load(int(t) for t in y)
# d=[]
# d.append(names)
# d.append(names2)
# d.append(y2)

# with open('models.yaml', 'a') as file:
#     yaml.dump_all(d, file, default_flow_style=False)
# with open('models.yaml', 'r') as file:
#     docs = yaml.safe_load_all(file)

#     for doc in docs:
#         print(doc)
import yaml

yaml_file = open("models.yaml", 'r')
yaml_content = yaml.load(yaml_file)

print("Key: Value")
for key, value in yaml_content.items():
    print(f"{key}: {value}")