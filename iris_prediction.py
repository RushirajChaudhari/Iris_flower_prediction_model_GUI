from tkinter import * 
from tkinter import messagebox
from sklearn.datasets import load_iris 
from sklearn.tree import DecisionTreeClassifier
import joblib

iris = load_iris()

X = iris.data
y = iris.target

model = DecisionTreeClassifier()
model.fit(X,y)

joblib.dump(model, "iris_model.joblib")

root = Tk()
root.title("Iris Flower Predictor using ML")
root.geometry("300x500")


label1 = Label(root , text="Sepal Length" , font=("Arial",20 , "bold"))
label1.pack( pady=10)
entry1 = Entry(root, font=("Arial", 14) , bg="gray", fg="white", width=10 , borderwidth=3)
entry1.pack(pady=10)


label2 = Label(root , text="Sepal Width" , font=("Arial",20 , "bold"))
label2.pack( pady=10)
entry2 = Entry(root, font=("Arial", 14) , bg="gray", fg="white", width=10 , borderwidth=3)
entry2.pack(pady=10)


label3 = Label(root , text="Petal Length" , font=("Arial",20 , "bold"))
label3.pack(pady=10)
entry3 = Entry(root, font=("Arial", 14) , bg="gray", fg="white", width=10 , borderwidth=3)
entry3.pack(pady=10)


label4 = Label(root , text="Sepal Width" , font=("Arial",20 , "bold"))
label4.pack(pady=10)
entry4 = Entry(root, font=("Arial", 14) , bg="gray", fg="white", width=10 , borderwidth=3)
entry4.pack(pady=10)
##########################################
species_label = Label(root, text="", bg="lightblue" , foreground="black" , font=("Arial", 15 , "bold"))
species_label.pack(pady=10)

# load our model
model = joblib.load("iris_model.joblib")

def predict_species():
    sepal_length = float(entry1.get())
    sepal_width = float(entry2.get())
    petal_length = float(entry3.get())
    petal_width = float(entry4.get())
    
    prediction = model.predict([[sepal_length,sepal_width,petal_length,petal_width]])
    species_label.config(text="Predicted Species: " +iris.target_names[prediction[0]])

my_button = Button(text="Predict" , bg="lightgreen", activebackground="blue", borderwidth=3, font=("Arial", 11 , "bold"), command=predict_species)
my_button.pack()

root.mainloop()