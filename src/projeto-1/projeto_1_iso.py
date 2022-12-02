# %%

import os
import json
import numpy as np
import pandas as pd

# %%

dimensions = open(os.path.join(os.path.dirname(__file__), "dimensions_projeto_1.json"), "r")
dimensions_data = json.load(dimensions)["dimensions"]
dimensions.close()

# %%

# dados da estrutura

# tensão admissível (Pa)
experimental_data = pd.read_csv(os.path.join(os.path.dirname(__file__), "Sigma_adm_experimentos.csv"), header=None)
mean = experimental_data[0].mean()
std_dv = experimental_data[0].std()

sig_adm = (mean - 2 * std_dv) * 10 ** 6

# massa específica (kg / m3)
p = 750

# módulo de elasticidade do material (Pa)
E = 15 * (10**9)

# comprimento: pilar esquerdo | viga superior | pilar direito (m)
Lpe, Lvs, Lpd = 4, 6, 6

# carga distribuída (kN / m)
q = 20

# força horizontal (kN)
Fh = 40

# %%


def beam_bending_moment(v, h, position):
    return (v * position) + (4 * h) - (((q / 2) * 10**3) * (position**2))


# %%


def calculate_structure(b1, b2, h2, b3):
    A1 = b1 * b1
    A2 = b2 * h2
    A3 = b3 * b3
    I1 = (b1 * (b1**3)) / 12
    I2 = (b2 * (h2**3)) / 12
    I3 = (b3 * (b3**3)) / 12

    Ha = (Fh * 10**3)

    Vb = ((((q / 2) * (Lvs ** 2)) + (4 * Fh)) / 6) * 10 ** 3
    Va = (Lvs * (q * 10 ** 3)) - Vb

    N1 = -Va
    M1 = 4 * Ha

    N2 = Ha - (Fh * 10**3)
    Me = beam_bending_moment(Va, Ha, 0)
    Mmiddle_beam = beam_bending_moment(Va, Ha, (Va / (q * 10**3)))
    Md = beam_bending_moment(Va, Ha, Lvs)
    Mmax = max([abs(Me), abs(Mmiddle_beam), abs(Md)])

    N3 = -Vb

    sig1 = (abs(N1) / A1) + ((abs(M1) / I1) * (b1 / 2))

    sig2 = (abs(N2) / A2) + ((abs(Mmax) / I2) * (h2 / 2))

    sig3 = (abs(N3) / A3)

    S = np.zeros((3, 1))
    fails = np.zeros((3, 1))
    tensions = [sig1, sig2, sig3]
    for i in range(len(tensions)):
        S[i][0] = tensions[i]

    for i in range(len(S)):
        if S[i][0] > sig_adm:
            fails[i][0] = 1

    volumes = [A1 * Lpe, A2 * Lvs, A3 * Lpd]
    weight = 0
    for volume in volumes:
        weight = weight + volume * p

    result_list = [
        b1,
        b2,
        h2,
        b3,
        int(weight),
        round(N1, 2),
        round(N2, 2),
        round(N3, 2),
        round(M1, 2),
        round(Mmax, 2),
        0,
        round(sig1, 2),
        round(sig2, 2),
        round(sig3, 2),
    ]

    if [1] in fails:
        return ["failed"]
    else:
        if abs(Mmiddle_beam) > abs(Me) and abs(Mmiddle_beam) > abs(Md):
            result_list.append("middle_beam")
            return result_list
        else:
            result_list.append("edge")
            return result_list


# %%


def get_lightweight_structure(structure_info):
    lightweight = min([structure[4] for structure in structure_info])

    lighter = []

    for structure in structure_info:
        if structure[4] == lightweight:
            lighter.append(structure)

    return lighter[0]


# %%

results = []

for dimension in dimensions_data:
    results.append(
        calculate_structure(dimension[0], dimension[1], dimension[2], dimension[3])
    )

# %%

not_failed = [result for result in results if result[0] != "failed"]

max_on_middle = [not_fail for not_fail in not_failed if not_fail[-1] == "middle_beam"]

# %%

optimized_options = {
    "middle_beam": get_lightweight_structure(max_on_middle)
}

# %%


def visualize_data(database):
    columns = ["b (m)", "h (m)", "Massa (kg)", "N (kN)", "Mmáx (kN.m)", "σ máx (MPa)"]

    firstRow, firstRowIndexes = [], [0, 4, 5, 8, 11]
    secondRow, secondRowIndexes = [], [1, 2, 4, 6, 9, 12]
    thirdRow, thirdRowIndexes = [], [3, 4, 7, 10, 13]

    for i in range(len(database)):
        if i in firstRowIndexes:
            if i == 0:
                firstRow.append(database[i])
                firstRow.append(database[i])
            else:
                firstRow.append(database[i])
        if i in secondRowIndexes:
            secondRow.append(database[i])
        if i in thirdRowIndexes:
            if i == 3:
                thirdRow.append(database[i])
                thirdRow.append(database[i])
            else:
                thirdRow.append(database[i])

    rows = [firstRow, secondRow, thirdRow]

    df = pd.DataFrame(rows)
    df.columns = columns
    df["σ máx / σ adm"] = df[columns[-1]] / sig_adm
    df[columns[-1]] = df[columns[-1]] / 10**6
    df[columns[3]] = df[columns[3]] / 10**3
    df[columns[4]] = df[columns[4]] / 10**3
    df.index = [1, 2, 3]
    df.index.name = "Elemento"

    return df

# %%

print("Momento máximo no vão da viga")
print(visualize_data(optimized_options["middle_beam"]))

# %%
