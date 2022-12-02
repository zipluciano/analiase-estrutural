import os

base_dir = os.path.dirname(__file__)

perfis = os.path.join(base_dir, "PerfisLista.py")
estrutura = os.path.join(base_dir, "projeto-2_clean.py")

for i in range(3):
    os.system(f"python3 {perfis}\n")
    os.system(f"python3 {estrutura}")