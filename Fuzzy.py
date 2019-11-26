import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt

x_pm25 = np.arange(0, 400, 1)
x_pm10 = np.arange(0, 40, 1)
x_humidity = np.arange(0, 100, 1)
x_cleaner = np.arange(0, 30, 1)

pm25_lo = fuzz.trapmf(x_pm25, [0, 0, 100, 200])
pm25_md = fuzz.trapmf(x_pm25, [160, 200, 250, 300])
pm25_hi = fuzz.trapmf(x_pm25, [270, 310, 400, 400])
pm10_lo = fuzz.trapmf(x_pm10, [0, 0, 12.5, 25])
pm10_hi = fuzz.trapmf(x_pm10, [20, 30, 40, 40])
humidity_lo = fuzz.trapmf(x_humidity, [0, 0, 15, 30])
humidity_md = fuzz.trapmf(x_humidity, [25, 40, 50, 60])
humidity_hi = fuzz.trapmf(x_humidity, [50, 70, 100, 100])

cleaner_lo = fuzz.trapmf(x_cleaner, [0, 0, 5, 12])
cleaner_md = fuzz.trapmf(x_cleaner, [0, 7, 20, 25])
cleaner_hi = fuzz.trapmf(x_cleaner, [20, 23, 30, 30])

fig, (ax0, ax1, ax2, ax3) = plt.subplots(nrows=4, figsize=(8, 9))

ax0.plot(x_pm25, pm25_lo, 'r', linewidth=2, label='niska')
ax0.plot(x_pm25, pm25_md, 'y', linewidth=2, label='srednia')
ax0.plot(x_pm25, pm25_hi, 'g', linewidth=2, label='Wysoka')
ax0.set_title('pm2.5 [µg/m3]')
ax0.legend()

ax1.plot(x_pm10, pm10_lo, 'r', linewidth=2, label='Niska')
ax1.plot(x_pm10, pm10_hi, 'y', linewidth=1.5, label='Wysoka')
ax1.set_title('pm10 [µg/m3]')
ax1.legend()

ax2.plot(x_humidity, humidity_lo, 'r', linewidth=2, label='Niska')
ax2.plot(x_humidity, humidity_md, 'y', linewidth=2, label='Srednia')
ax2.plot(x_humidity, humidity_hi, 'g', linewidth=2, label='Wysoka')
ax2.set_title('Wilgotność [%]')
ax2.legend()

ax3.plot(x_cleaner, cleaner_lo, 'r', linewidth=2, label='Niska')
ax3.plot(x_cleaner, cleaner_md, 'y', linewidth=2, label='Srednia')
ax3.plot(x_cleaner, cleaner_hi, 'g', linewidth=2, label='Wysoka')
ax3.set_title('Obroty')
ax3.legend()

# Turn off top/right axes
for ax in (ax0, ax1, ax2, ax3):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

plt.tight_layout()
plt.show()

pm25_level_lo = fuzz.interp_membership(x_pm25, pm25_lo, 300)
pm25_level_md = fuzz.interp_membership(x_pm25, pm25_md, 300)
pm25_level_hi = fuzz.interp_membership(x_pm25, pm25_hi, 300)

pm10_level_lo = fuzz.interp_membership(x_pm10, pm10_lo, 25)
pm10_level_hi = fuzz.interp_membership(x_pm10, pm10_hi, 25)

humidity_level_lo = fuzz.interp_membership(x_humidity, humidity_lo, 60)
humidity_level_md = fuzz.interp_membership(x_humidity, humidity_md, 60)
humidity_level_hi = fuzz.interp_membership(x_humidity, humidity_hi, 60)

active_rule1 = np.fmin(pm25_level_lo, np.fmin(pm10_level_lo, humidity_level_lo))

cleaner_activation_lo = np.fmin(active_rule1, cleaner_lo)

cleaner_activation_md = np.fmin(pm10_level_lo, cleaner_lo)

active_rule3 = np.fmin(pm25_level_hi, pm10_level_hi)
cleaner_activation_hi = np.fmin(active_rule3, cleaner_hi)
cleaner0 = np.zeros_like(x_cleaner)

fig, ax0 = plt.subplots(figsize=(8, 3))

ax0.fill_between(x_cleaner, cleaner0, cleaner_activation_lo, facecolor='b', alpha=0.7)
ax0.plot(x_cleaner, cleaner_lo, 'r', linewidth=2, linestyle='--', )
ax0.fill_between(x_cleaner, cleaner0, cleaner_activation_md, facecolor='g', alpha=0.7)
ax0.plot(x_cleaner, cleaner_md, 'y', linewidth=2, linestyle='--')
ax0.fill_between(x_cleaner, cleaner0, cleaner_activation_hi, facecolor='r', alpha=0.7)
ax0.plot(x_cleaner, cleaner_hi, 'g', linewidth=2, linestyle='--')
ax0.set_title('Wyjściowe funkcje aktywacji')

# Turn off top/right axes
for ax in (ax0,):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

plt.tight_layout()
plt.show()

aggregated = np.fmax(cleaner_activation_lo,
                     np.fmax(cleaner_activation_md, cleaner_activation_hi))

cleaner = fuzz.defuzz(x_cleaner, aggregated, 'som')
cleaner_activation = fuzz.interp_membership(x_cleaner, aggregated, cleaner)  # for plot

# Visualize this
fig, ax0 = plt.subplots(figsize=(8, 3))

ax0.plot(x_cleaner, cleaner_lo, 'r', linewidth=2, linestyle='--', )
ax0.plot(x_cleaner, cleaner_md, 'y', linewidth=2, linestyle='--')
ax0.plot(x_cleaner, cleaner_hi, 'g', linewidth=2, linestyle='--')
ax0.fill_between(x_cleaner, cleaner0, aggregated, facecolor='Orange', alpha=0.7)
ax0.plot([cleaner, cleaner], [0, cleaner_activation], 'k', linewidth=1.5, alpha=0.9)
ax0.set_title('Wynik')

# Turn off top/right axes
for ax in (ax0,):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

plt.tight_layout()
plt.show()
