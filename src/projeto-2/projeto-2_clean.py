import os
import numpy as np
import pandas as pd
import json

# Dados do problema "Pórticos: EXEMPLO 02" da lista

n_nos = 12  # Número de nós
n_el = 15  # Número de elementos
a = 0.3
# número de cada nó e coordenadas x e y dos mesmos
no = [i for i in range(12)]
x = [
    2 * 30 * a,
    0,
    0,
    0,
    0,
    30 * a,
    30 * a,
    30 * a,
    30 * a,
    2 * 30 * a,
    2 * 30 * a,
    2 * 30 * a,
]

y = [
    3 * 12 * a,
    0,
    12 * a,
    2 * 12 * a,
    3 * 12 * a,
    0,
    12 * a,
    2 * 12 * a,
    3 * 12 * a,
    0,
    12 * a,
    2 * 12 * a,
]

# %%
dimensions = open(os.path.join(os.path.dirname(__file__), "dimensions_projeto_2.json"), "r")
dimensions_data = json.load(dimensions)["dimensions"]
dimensions_data = np.array(dimensions_data)
dimensions.close()

# %%
# [p1, p2, p3, p4, p5, p6, v1, v2, v3]
# Matriz de Seções: [número da seção, área, módulo de elasticidade, momento de inércia,
def calcula_estrutura(dimension):
    untouched_dimensions = dimension.copy()
    secoes = dimension
    n_sec = len(secoes)  # Número de seções distintas presentes na estrutura

    for i in range(n_sec):
        secoes[i, 0] = i

    # Matriz de conectividade: [elemento, Número da seção, primeiro nó, segundo nó]
    conec = np.array(
        [
            [0, int(secoes[-1, 0]), 8, 0],
            [1, int(secoes[0, 0]), 1, 2],
            [2, int(secoes[3, 0]), 5, 6],
            [3, int(secoes[0, 0]), 9, 10],
            [4, int(secoes[1, 0]), 2, 3],
            [5, int(secoes[4, 0]), 6, 7],
            [6, int(secoes[1, 0]), 10, 11],
            [7, int(secoes[2, 0]), 3, 4],
            [8, int(secoes[5, 0]), 7, 8],
            [9, int(secoes[2, 0]), 11, 0],
            [10, int(secoes[-3, 0]), 2, 6],
            [11, int(secoes[-3, 0]), 6, 10],
            [12, int(secoes[-2, 0]), 3, 7],
            [13, int(secoes[-2, 0]), 7, 11],
            [14, int(secoes[-1, 0]), 4, 8],
        ]
    )

    # %%
    # Carregamentos nodais (Fzão da estrutura)
    n_forcas = 3  # Número de nós na qual atuam forças
    # Matriz de forças [nó (primeiro nó é o nó zero e não 1), força em x, força em y, momento]
    forca_nodal = 30 * (10**3)
    forcas = np.matrix(
        [[2, forca_nodal, 0, 0], [3, forca_nodal, 0, 0], [4, forca_nodal, 0, 0]]
    )

    # %%
    # Carregamentos equivalentes (Feq da estrutura)
    n_eq = 6  # número de elementos que contem carregamentos equivalentes
    # Matriz de carregamento equivalente = [elemento, tipo de carregamento, intensidade, posição (para o caso de carregamento concentrado entre nós)]
    carreg_uniforme = 18 * (10**3)
    w_eq = np.array(
        [
            [10, 1, -carreg_uniforme, 0],
            [11, 1, -carreg_uniforme, 0],
            [12, 1, -carreg_uniforme, 0],
            [13, 1, -carreg_uniforme, 0],
            [14, 1, -carreg_uniforme, 0],
            [0, 1, -carreg_uniforme, 0],
        ]
    )
    # LEMBRETE: os sinais das forças devem seguir o sistema LOCAL do elemento!

    # %%
    # Apoios
    n_rest = 3  # número de nós restringidos
    # Matriz de condições de contorno
    # [número do nó, restringido_x, restringido_y, restringido_theta] (1 para restringido, e 0 para livre)
    GDL_rest = np.array([[1, 1, 1, 0], [5, 1, 1, 0], [9, 1, 1, 0]])

    # %%
    # CALCULO DA ESTRUTURA
    GDL = 3 * n_nos  # graus de liberdade da estrutura
    K = np.zeros((GDL, GDL))  # matriz rigidez global

    # Cálculo da matriz de cada elemento
    for el in range(n_el):
        # print(el)
        # calculo do comprimento do elemento el
        no1 = conec[el, 2]
        no2 = conec[el, 3]
        # L=abs(x(no2)-x(no1))
        L = np.sqrt((x[no2] - x[no1]) ** 2 + (y[no2] - y[no1]) ** 2)
        # Propriedades
        A = secoes[conec[el, 1], 1]
        E = secoes[conec[el, 1], 2]
        Iz = secoes[conec[el, 1], 3]
        # Cossenos diretores a partir das coordenadas dos ns do elemento
        c = (x[no2] - x[no1]) / L  # cosseno
        s = (y[no2] - y[no1]) / L  #  seno
        # Matriz de transformação do elemento "el"
        T = np.array(
            [
                [c, s, 0, 0, 0, 0],
                [-s, c, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, c, s, 0],
                [0, 0, 0, -s, c, 0],
                [0, 0, 0, 0, 0, 1],
            ]
        )
        # Construo da matriz de rigidez em coordenadas locais
        k1 = E * A / L
        k2 = 12 * E * Iz / L**3
        k3 = 6 * E * Iz / L**2
        k4 = 4 * E * Iz / L
        k5 = k4 / 2
        k = np.array(
            [
                [k1, 0, 0, -k1, 0, 0],
                [0, k2, k3, 0, -k2, k3],
                [0, k3, k4, 0, -k3, k5],
                [-k1, 0, 0, k1, 0, 0],
                [0, -k2, -k3, 0, k2, -k3],
                [0, k3, k5, 0, -k3, k4],
            ]
        )
        # Matriz de rigidez em coordenadas globais
        kg = np.dot(np.transpose(T), np.dot(k, T))

        # Determinando matriz de incidência cinemática:
        b = np.zeros((6, GDL))
        i = no1
        j = no2
        b[0, 3 * i] = 1
        b[1, 3 * i + 1] = 1
        b[2, 3 * i + 2] = 1
        b[3, 3 * j] = 1
        b[4, 3 * j + 1] = 1
        b[5, 3 * j + 2] = 1
        # Expandindo e convertendo a matriz do elemento para coordenadas globais:
        Ki = np.dot(np.transpose(b), np.dot(kg, b))
        # Somando contribuição do elemento para a matriz de rigidez global:
        K = K + Ki

    # %%
    # Vetor de forcas Global
    F = np.zeros((GDL, 1))
    for i in range(n_forcas):
        F[int(3 * forcas[i, 0])] = forcas[i, 1]
        F[int(3 * forcas[i, 0]) + 1] = forcas[i, 2]
        F[int(3 * forcas[i, 0]) + 2] = forcas[i, 3]

    # %%
    # Construção do vetor de foras equivalentes
    Feq = np.zeros((GDL, 1))
    for i in range(n_eq):
        tipo = int(w_eq[i, 1])  # tipo de força equivalente
        el = int(w_eq[i, 0])  # elemento onde está aplicada
        if tipo == 1:  # Carregamento distribuído
            f = np.zeros((6, 1))
            no1 = conec[el, 2]
            no2 = conec[el, 3]
            L = np.sqrt((x[no2] - x[no1]) ** 2 + (y[no2] - y[no1]) ** 2)
            w = w_eq[i, 2]
            f[0] = 0
            f[1] = +w * L / 2
            f[2] = +w * L**2 / 12
            f[3] = 0
            f[4] = +w * L / 2
            f[5] = -w * L**2 / 12
            # Cossenos diretores a partir das coordenadas dos ns do elemento
            c = (x[no2] - x[no1]) / L  # cosseno
            s = (y[no2] - y[no1]) / L  #  seno
            # Matriz de transformação do elemento "el"
            T = np.array(
                [
                    [c, s, 0, 0, 0, 0],
                    [-s, c, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0],
                    [0, 0, 0, c, s, 0],
                    [0, 0, 0, -s, c, 0],
                    [0, 0, 0, 0, 0, 1],
                ]
            )
            # feqTT=np.dot(np.transpose(T),f)
            feq = np.matmul(np.transpose(T), f)
            Feq[3 * no1] = Feq[3 * no1] + feq[0]
            Feq[3 * no1 + 1] = Feq[3 * no1 + 1] + feq[1]
            Feq[3 * no1 + 2] = Feq[3 * no1 + 2] + feq[2]
            Feq[3 * no2] = Feq[3 * no2] + feq[3]
            Feq[3 * no2 + 1] = Feq[3 * no2 + 1] + feq[4]
            Feq[3 * no2 + 2] = Feq[3 * no2 + 2] + feq[5]
        elif tipo == 2:  ## carga aplicada a uma distancia a do nó i
            f = np.zeros((6, 1))
            no1 = conec[el, 2]
            no2 = conec[el, 3]
            L = np.sqrt((x[no2] - x[no1]) ** 2 + (y[no2] - y[no1]) ** 2)
            a = w_eq[i, 3]
            b = L - a
            p = w_eq[i, 2]
            f[0] = 0
            f[1] = +p * b**2 * (3 * a + b) / L**3
            f[2] = +p * a * b**2 / L**2
            f[3] = 0
            f[4] = +p * a**2 * (a + 3 * b) / L**3
            f[5] = -p * a**2 * b / L**2
            # Cossenos diretores a partir das coordenadas dos nós do elemento
            c = (x[no2] - x[no1]) / L  # cosseno
            s = (y[no2] - y[no1]) / L  #  seno
            # Matriz de transformação do elemento "el"
            T = np.array(
                [
                    [c, s, 0, 0, 0, 0],
                    [-s, c, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0],
                    [0, 0, 0, c, s, 0],
                    [0, 0, 0, -s, c, 0],
                    [0, 0, 0, 0, 0, 1],
                ]
            )
            # feqTT=np.dot(np.transpose(T),f)
            feq = np.matmul(np.transpose(T), f)
            Feq[3 * no1] = Feq[3 * no1] + feq[0]
            Feq[3 * no1 + 1] = Feq[3 * no1 + 1] + feq[1]
            Feq[3 * no1 + 2] = Feq[3 * no1 + 2] + feq[2]
            Feq[3 * no2] = Feq[3 * no2] + feq[3]
            Feq[3 * no2 + 1] = Feq[3 * no2 + 1] + feq[4]
            Feq[3 * no2 + 2] = Feq[3 * no2 + 2] + feq[5]

    # %%
    # guardamos os originais de K e F
    Kg = np.copy(K)
    # Kg[:] = K[:]

    Fg = F + Feq
    # Aplicar Restrições (condições de contorno)
    for k in range(n_rest):
        # Verifica se há restrição na direção x
        if GDL_rest[k, 1] == 1:
            j = 3 * GDL_rest[k, 0]
            # Modificar Matriz de Rigidez
            for i in range(GDL):
                Kg[j, i] = 0  # zera linha
                Kg[i, j] = 0  # zera coluna
            Kg[j, j] = 1  # valor unitário na diagonal principal
            Fg[j] = 0
        # Verifica se há restrição na direção y
        if GDL_rest[k, 2] == 1:
            j = 3 * GDL_rest[k, 0] + 1
            # Modificar Matriz de Rigidez
            for i in range(GDL):
                Kg[j, i] = 0  # zera linha
                Kg[i, j] = 0  # zera coluna
            Kg[j, j] = 1  # valor unitário na diagonal principal
            Fg[j] = 0
        # Verifica se há restrição na rotação
        if GDL_rest[k, 3] == 1:
            j = 3 * GDL_rest[k, 0] + 2
            # Modificar Matriz de Rigidez
            for i in range(GDL):
                Kg[j, i] = 0  # zera linha
                Kg[i, j] = 0  # zera coluna
            Kg[j, j] = 1  # valor unitário na diagonal principal
            Fg[j] = 0

    # %%
    # Calculo dos deslocamentos
    desloc = np.linalg.solve(Kg, Fg)

    # %%
    # Reações
    reacoes = np.matmul(K, desloc) - Feq
    # reacoes=K*desloc-Feq

    # %%
    # Esforços nos elementos
    f_el = np.zeros((n_el, 6))
    N = np.zeros((n_el, 1))
    Mmax = np.zeros((n_el, 1))
    Smax = np.zeros((n_el, 1))
    Falha = np.zeros((n_el, 1))
    Sadm = 248.2e6  # colocar valor do pdf do projeto
    peso = 0
    ro = 7861
    for el in range(n_el):
        # calculo do comprimento do elemento el
        no1 = conec[el, 2]
        no2 = conec[el, 3]
        # L=abs(x(no2)-x(no1))
        L = np.sqrt((x[no2] - x[no1]) ** 2 + (y[no2] - y[no1]) ** 2)
        # Propriedades
        A = secoes[conec[el, 1], 1]
        E = secoes[conec[el, 1], 2]
        Iz = secoes[conec[el, 1], 3]
        cc = secoes[conec[el, 1], 4]
        # calculo peso
        peso = peso + A * L * ro
        # Cossenos diretores a partir das coordenadas dos ns do elemento
        c = (x[no2] - x[no1]) / L  # cosseno
        s = (y[no2] - y[no1]) / L  #  seno
        # Matriz de transformação do elemento "el"
        T = np.array(
            [
                [c, s, 0, 0, 0, 0],
                [-s, c, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, c, s, 0],
                [0, 0, 0, -s, c, 0],
                [0, 0, 0, 0, 0, 1],
            ]
        )
        # Construção da matriz de rigidez em coordenadas locais
        k1 = E * A / L
        k2 = 12 * E * Iz / L**3
        k3 = 6 * E * Iz / L**2
        k4 = 4 * E * Iz / L
        k5 = k4 / 2
        ke = np.array(
            [
                [k1, 0, 0, -k1, 0, 0],
                [0, k2, k3, 0, -k2, k3],
                [0, k3, k4, 0, -k3, k5],
                [-k1, 0, 0, k1, 0, 0],
                [0, -k2, -k3, 0, k2, -k3],
                [0, k3, k5, 0, -k3, k4],
            ]
        )
        # pega os valores dos deslocamentos dos nós do elemento "el"
        u1 = desloc[no1 * 3]
        u2 = desloc[no2 * 3]
        v1 = desloc[no1 * 3 + 1]
        v2 = desloc[no2 * 3 + 1]
        th1 = desloc[no1 * 3 + 2]
        th2 = desloc[no2 * 3 + 2]
        d_g = np.array([u1, v1, th1, u2, v2, th2])
        d_el = np.matmul(T, d_g)
        # d_el=T*d_g

        ## forças equivalentes: recalcula vetor de feq. no sistema local
        aux = []
        cont = [0]
        for temp in w_eq[:, 0]:
            if int(temp) == el:
                aux = cont[:]
            cont[0] = cont[0] + 1
        if len(aux) == 0:
            feqq = 0
        else:
            aux = int(aux[0])
            tipo = w_eq[aux, 1]  # tipo de força equivalente
            if tipo == 1:
                w = w_eq[aux, 2]
                feqq = np.zeros((6, 1))
                feqq[0] = 0
                feqq[1] = +w * L / 2
                feqq[2] = +w * L**2 / 12
                feqq[3] = 0
                feqq[4] = +w * L / 2
                feqq[5] = -w * L**2 / 12
            elif tipo == 2:
                a = w_eq[aux, 3]
                b = L - a
                p = w_eq[aux, 2]
                feqq = np.zeros((6, 1))
                feqq[0] = 0
                feqq[1] = +p * b**2 * (3 * a + b) / L**3
                feqq[2] = +p * a * b**2 / L**2
                feqq[3] = 0
                feqq[4] = +p * a**2 * (a + 3 * b) / L**3
                feqq[5] = -p * a**2 * b / L**2

        ## esforços locais atuantes no elemento "el": cada linha da matriz f_el
        # contem os esforços de um elemento = [fx_1' fy_1' mz_1' fx_2' fy_2' mz_2']
        f_el[el, :] = np.transpose(np.matmul(ke, d_el) - feqq)
        # Esforços para cálculo de tensão
        N = abs(f_el[el, 0])
        Mzi = abs(f_el[el, 2])
        Mzj = abs(f_el[el, 5])
        if el > 0 and el < 10:
            aux = np.array([Mzi, Mzj])
            Mmax[el] = aux.max()
        else:
            Mvao = -f_el[el, 2] + f_el[el, 1] / (-2 * w)
            aux = np.array([Mzi, Mzj, Mvao])
            Mmax[el] = aux.max()

        # Cálculo da tensão
        Smax[el] = N / A + Mmax[el] / Iz * cc

        # Critério de Falha
        if Smax[el] > Sadm:
            Falha[el] = 1

    check_fail = np.where(Falha == [1])
    is_failed = len(check_fail[0]) > 0

    returned_object = {
        "N": [n[0] for n in f_el],
        "Mmax": Mmax,
        "Smax": Smax,
        "Peso": [peso for i in f_el],
        "Dimensoes": untouched_dimensions
    }

    if is_failed:
        return "Falhou"
    else:
        return returned_object

results = []
total_weight = []
optimal_structure = {}

for i in range(len(dimensions_data)):
    structure = calcula_estrutura(dimensions_data[i])
    if structure == "Falhou":
        pass
    else:
        results.append(structure)

for result in results:
    total_weight.append(result["Peso"][0])

minimum_weight = min(total_weight)

for result in results:
    if result["Peso"][0] == minimum_weight:
        optimal_structure = result
        break

rows_data_viz = [optimal_structure["Peso"],
                 optimal_structure["N"],
                 [m_max[0] for m_max in optimal_structure["Mmax"]],
                 [s_max[0] for s_max in optimal_structure["Smax"]]
                ]

df = pd.DataFrame(rows_data_viz).transpose()
columns = ["Peso (kg)", "N (kN)", "Mmáx (kN.m)", "σ máx (MPa)"]
df.index.name = "Elemento"
df.columns = columns
df[columns[0]] = abs(round(df[columns[0]], 2))
df[columns[1]] = abs(round(df[columns[1]] / (10 ** 3), 2))
df[columns[2]] = abs(round(df[columns[2]] / (10 ** 3), 2))
df[columns[3]] = abs(round(df[columns[3]] / (10 ** 6), 2))
df["%"] = abs(round((df[columns[2]] / 248.2) * 100, 2))

structure_weight = optimal_structure["Peso"][0]

print(df)