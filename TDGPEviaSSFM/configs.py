import matplotlib.pyplot as plt

plt.rcParams["figure.dpi"] = 80
plt.rcParams["figure.figsize"] = [15, 10]

dt = 0.005
N_t = 24_000

secsToMsecsConversionFactor = 1000
partFactor = 1
skippingFactor = 5
savePath = "/vagrant/animations/{furtherInfo['name']}.mp4"

