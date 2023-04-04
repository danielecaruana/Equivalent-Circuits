import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# Importing Data and defining vars

df_s11 = pd.read_excel('Data.xlsx', 0, header=1)
nacl = pd.read_excel('Data.xlsx', 1)
methanol = pd.read_excel('Data.xlsx', 2)
df_e3 = pd.read_excel('Data.xlsx', 3)

air_s11 = df_s11.iloc[:, 1].to_numpy() + (df_s11.iloc[:, 2].to_numpy() * 1j)
water_s11 = df_s11.iloc[:, 4].to_numpy() + (df_s11.iloc[:, 5].to_numpy() * 1j)
methanol_s11 = df_s11.iloc[:, 7].to_numpy() + (df_s11.iloc[:, 8].to_numpy() * 1j)
nacl_s11 = df_s11.iloc[:, 10].to_numpy() + (df_s11.iloc[:, 11].to_numpy() * 1j)
short_s11 = df_s11.iloc[:, 13].to_numpy()
e3_arr = df_e3.iloc[:, 1].to_numpy() + (df_e3.iloc[:, 2].to_numpy() * 1j)
frequency_arr = df_s11.iloc[:, 0].to_numpy()

exp_em_methanol = methanol.iloc[:, 1].to_numpy() + (methanol.iloc[:, 2].to_numpy() * 1j)
exp_em_nacl = nacl.iloc[:, 1].to_numpy() + (nacl.iloc[:, 2].to_numpy() * 1j)

# Defining reflection coefficients

methanol_d_m2 = methanol_s11 - air_s11
methanol_d_m1 = methanol_s11 - short_s11
methanol_d_m3 = methanol_s11 - water_s11
nacl_d_m2 = nacl_s11 - air_s11
nacl_d_m1 = nacl_s11 - short_s11
nacl_d_m3 = nacl_s11 - water_s11
d_13 = short_s11 - water_s11
d_21 = air_s11 - short_s11
d_32 = water_s11 - air_s11
e2 = 1

# Fitting the function - Methanol

methanol_em_calc = -(e3_arr * (methanol_d_m2 * d_13) / (methanol_d_m1 * d_32)) \
                   - (e2 * (methanol_d_m3 * d_21) / (methanol_d_m1 * d_32))

res_methanol_calc, cov_methanol_calc = np.polyfit(frequency_arr, methanol_em_calc, deg=5, cov=True)
p_methanol_calc = np.poly1d(res_methanol_calc)
methanol_extend = np.linspace(5e8, 1.2e10, 201)
tr_x_methanol_calc = p_methanol_calc(methanol_extend)

res_methanol_giv, cov_methanol_giv = np.polyfit(frequency_arr, exp_em_methanol, deg=5, cov=True)
p_methanol_giv = np.poly1d(res_methanol_giv)
me_giv_extend = np.linspace(5e8, 1.2e10, 201)
tr_x_methanol_giv = p_methanol_giv(me_giv_extend)

# Fitting the function - NaCl

nacl_em_calc = -(e3_arr * (nacl_d_m2 * d_13) / (nacl_d_m1 * d_32)) \
               - (e2 * (nacl_d_m3 * d_21) / (nacl_d_m1 * d_32))

res_nacl_calc, cov_nacl_calc = np.polyfit(frequency_arr, nacl_em_calc, deg=5, cov=True)
p_nacl_calc = np.poly1d(res_nacl_calc)
nacl_extend = np.linspace(5e8, 1.2e10, 201)
tr_x_nacl_calc = p_nacl_calc(nacl_extend)

res_nacl_giv, cov_nacl_giv = np.polyfit(frequency_arr, exp_em_nacl, deg=5, cov=True)
p_nacl_giv = np.poly1d(res_nacl_giv)
na_giv_extend = np.linspace(5e8, 1.2e10, 201)
tr_x_nacl_giv = p_nacl_giv(na_giv_extend)

# Plotting Graph - Methanol

rc = {"font.family": "serif",
      "mathtext.fontset": "stix"}
plt.rcParams.update(rc)
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
hfont = {'fontname': 'Times New Roman'}

fig_methanol, ax_methanol = plt.subplots(1)
ax_methanol.figure.set_size_inches(8.27, 11.69)


plt.scatter(frequency_arr, np.real(methanol_em_calc), color='black', marker='+', linewidths=0.7,
            label="Calculated values")
plt.plot(methanol_extend, np.real(tr_x_methanol_calc), color='black', label='Trendline for calculated values')

plt.scatter(frequency_arr, np.real(exp_em_methanol), marker='x', color='grey', linewidth=1, label='Given values')
plt.plot(me_giv_extend, np.real(tr_x_methanol_giv), linestyle='-.', color="black", label='Trendline for given values')

plt.grid(which='both', linestyle='--')
plt.minorticks_on()

plt.xlim(4e8, 1.02e10)
plt.title(r"A graph of $\varepsilon_{Methanol}$ / $\mathrm{F} \ \mathrm{m}^{-1}$ against $f$ / $\mathrm{Hz}$ for "
          r"Methanol")
plt.xlabel(r"$f$ / $\mathrm{Hz}$")
plt.ylabel(r"$\varepsilon_{Methanol}$ / $\mathrm{F} \ m^{-1}$")
formatter = mticker.ScalarFormatter(useMathText=True)
ax_methanol.xaxis.set_major_formatter(formatter)
plt.legend()
plt.savefig('methanol.pdf')
plt.show()

# Plotting graph - NaCl

fig_nacl, ax_nacl = plt.subplots(1)
ax_nacl.figure.set_size_inches(8.27, 11.69)

plt.scatter(frequency_arr, np.real(nacl_em_calc), color='black', marker='+', linewidths=0.7, label="Calculated values")
plt.plot(nacl_extend, np.real(tr_x_nacl_calc), color='black', label='Trendline for calculated values')

plt.scatter(frequency_arr, np.real(exp_em_nacl), marker='x', color='grey', linewidth=1, label='Given values')
plt.plot(na_giv_extend, np.real(tr_x_nacl_giv), linestyle='-.', color="black", label='Trendline for given values')

plt.grid(which='both', linestyle='--')
plt.minorticks_on()

plt.xlim(4e8, 1.02e10)

plt.title(r"A graph of $\varepsilon_{NaCl}$ / $\mathrm{F} \ \mathrm{m}^{-1}$ against $f$ / $\mathrm{Hz}$ for "
          r"NaCl")
plt.xlabel(r"$f$ / $\mathrm{Hz}$")
plt.ylabel(r"$\varepsilon_{NaCl}$ / $\mathrm{F} \ m^{-1}$")
formatter = mticker.ScalarFormatter(useMathText=True)
ax_nacl.xaxis.set_major_formatter(formatter)
plt.legend()
plt.savefig('nacl.pdf')
plt.show()

print("Debug")
