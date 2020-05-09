import matplotlib.pyplot as plt

plt.rcParams["figure.dpi"] = 80
plt.rcParams["figure.figsize"] = [15, 10]

dt = 0.005   # Erlaubt: 0.02, 0.01, 0.005
N_t = 24_000    # Wenn dt = 0.005, dann 200 für eine Sekunde, 12_000 für eine Minute.

secsToMsecsConversionFactor = 1000
partFactor = 1
skippingFactor = int(20 / (dt * 1000))
savePath = "/vagrant/animations/{furtherInfo['name']}.mp4"

