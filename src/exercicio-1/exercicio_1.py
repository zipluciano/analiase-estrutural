# %%

import numpy as np

# %% [markdown]

# <img src="./exercicio_1.png" />

# %%

# tensão admissível
sig_adm = 9.183 * (10 ** 6)

# massa específica
p = 750

# %%

# geometria das peças (m)
dimensions = [0.1, 0.15, 0.4, 0.2]
b1, b2, h2, b3 = dimensions[0], dimensions[1], dimensions[2], dimensions[3]

# geometria da estrutura (m)
# pilar lateral esquerdo
Lpe = 4
# pilar lateral direito
Lpd = 6
# viga superior
Lvs = 6

# cargas externas
# força horizontal (kN)
Fh = 40
# carga distribuída (kN/m)
q = 20

# %%

A1 = b1 * b1
A2 = b2 * h2
A3 = b3 * b3
I1 = (b1 * (b1 ** 3)) / 12
I2 = (b2 * (h2 ** 3)) / 12
I3 = (b3 * (b3 ** 3)) / 12
E1 = 15 * (10 ** 9)

# %%

def first_horizontal_reaction(b1, b2, h2, b3):
    A1 = b1 * b1
    A2 = b2 * h2
    A3 = b3 * b3
    I1 = (b1 * (b1 ** 3)) / 12
    I2 = (b2 * (h2 ** 3)) / 12
    I3 = (b3 * (b3 ** 3)) / 12
    E1 = 15 * (10 ** 9)

    constants = [
        [2560, 3 * E1 * I1],
        [4040, E1 * I2],
        [400, 9 * E1 * A1],
        [-520, 3 * E1 * A3],
        [64, 3 * E1 * I1],
        [152, E1 * I2],
        [72, E1 * I3],
        [4, 9 * E1 * A1],
        [6, E1 * A2],
        [2, 3 * E1 * A3],
    ]

    up = 0
    down = 0

    for i in range(len(constants)):
        if i <= 3:
            up = up + constants[i][0] / constants[i][1]
        else:
            down = down + constants[i][0] / constants[i][1]

    Hb = up / down

    return Hb * 10 ** 3

# %%

# Cálculo da reação Hb

Hb = first_horizontal_reaction(b1, b2, h2, b3)

print("Hb: {:+.2f} kN".format(Hb / 10 ** 3))

# %%

Ha = (Fh * 10 ** 3) - Hb
print("Ha: {:+.2f} kN".format(Ha / 10 ** 3))

# %%

Va = (((Lpd - Lpe) * Ha) + ((q * 10 ** 3) * (Lvs ** 2) / 2) + (-(Fh * 10 ** 3) * Lpd)) / 6
print("Va: {:+.2f} kN".format(Va / 10 ** 3))

# %%

Vb = ((q * 10 ** 3) * Lvs) - Va
print("Vb: {:+.2f} kN".format(Vb / 10 ** 3))

# %%
# esforços no pilar esquerdo
N1 = abs(Va)
M1 = abs(4 * Ha)
print("N1: {:+.2f} N".format(N1))
print("M1: {:+.2f} N.m".format(M1))

# %%

# esforços na viga superior
def left_bending_moment(v, h, position):
    return (v * position) + (4 * h) - (((q / 2) * 10 ** 3) * (position ** 2))


N2 = Ha - (Fh * 10 ** 3)
Me = left_bending_moment(Va, Ha, 0)
Mvao = left_bending_moment(Va, Ha, (Va / (q * 10 ** 3)))
Md = left_bending_moment(Va, Ha, 6)
Mmax = max([abs(Me), abs(Mvao), abs(Md)])

print("N2: {:+.2f} N".format(N2))
print("Me: {:+.2f} N.m".format(Me))
print("Mvao: {:+.2f} N.m".format(Mvao))
print("Md: {:+.2f} N.m".format(Md))
print("Mmax: {:+.2f} N.m".format(Mmax))

# %%

# esforços no pilar direito
N3 = -Vb
M3 = 6 * Hb
print("N3: {:+.2f} N".format(N3))
print("M3: {:+.2f} N.m".format(M3))

# %%

sig_adm = 5 * (10 ** 6)

# tensões internas
sig1 = ((abs(N1) / A1) + ((abs(M1) / I1) * (b1 / 2)))
print("sig1: {:+.2f} MPa".format(sig1 / 10 ** 6))

sig2 = ((abs(N2) / A2) + ((abs(Mmax) / I2) * (h2 / 2)))
print("sig2: {:+.2f} MPa".format(sig2 / 10 ** 6))

sig3 = ((abs(N3) / A3) + ((abs(M3) / I3) * (b3 / 2)))
print("sig3: {:+.2f} MPa".format(sig3 / 10 ** 6))

# %%

S = np.zeros((3, 1))
fails = np.zeros((3, 1))
tensions = [sig1, sig2, sig3]
for i in range(len(tensions)):
    S[i][0] = tensions[i]

for i in range(len(S)):
    if S[i][0] > sig_adm:
        fails[i][0] = 1

# %%

volumes = [A1 * Lpe, A2 * Lvs, A3 * Lpd]
weights = [i * p for i in volumes]

# %%
