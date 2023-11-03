import matplotlib.pyplot as plt
import pandas as pd

results =  pd.read_csv("outputs/asr_batch_32.csv")

fig1 = plt.figure("Figure 1")

plt.plot(results["epoch"], results["loss"], results["val_loss"])
plt.legend(["loss", "val_loss"])
plt.xlabel("epoch")
plt.ylabel("loss")
plt.title("model train vs. validation loss")
plt.savefig("outputs/loss.jpg")

fig2 = plt.figure("Figure 2")

plt.plot(results["epoch"], results["word_error_rate"])
plt.xlabel("epoch")
plt.ylabel("word_error_rate")
plt.title("validation word error rate")
plt.savefig("outputs/word_error_rate.jpg")
