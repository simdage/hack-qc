
import matplotlib.pyplot as plt

plt.figure()
plt.title("Anomalie flagrante pour la série Beauharnois - Aval")
plt.plot(df["Brutte_aval"])
plt.scatter(x = df[df["Brutte_aval"] < 20].index, y = df[df["Brutte_aval"] < 20]["Brutte_aval"], color = "red")
plt.scatter(x = df[df["Brutte_aval"] > 24].index, y = df[df["Brutte_aval"] > 24]["Brutte_aval"], color = "red")
plt.ylabel("Niveau")
plt.savefig("anomalie_flagrante.png")
plt.show()