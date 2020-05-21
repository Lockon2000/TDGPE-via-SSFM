import matplotlib.pyplot as plt

plt.rcParams["figure.dpi"] = 80
plt.rcParams["figure.figsize"] = [15, 15]

dt = 0.01
N_t = 24_000

skippingFactor = 16
secsToMsecsConversionFactor = 0.125 * 1000
partFactor = 1
savePath = "/vagrant/animations/{furtherInfo['name']}.mp4"

unitSystem = "natural"
smoothingParameters = [15,2]
