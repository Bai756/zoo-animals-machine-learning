import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import tkinter as tk
from tkinter import OptionMenu, StringVar, ttk

def predict(animal, model):
    
    database = pd.read_csv("zoo.csv")
    classes = pd.read_csv("class.csv", usecols=['Class_Number', 'Class_Type'])
    animal_values = database[database['animal_name'] == animal].copy()
    animal_values.drop(columns=['animal_name', 'class_type'], inplace=True)
    animal_values = animal_values.values

    prediction = model.predict(animal_values)
    prediction = prediction[0]
    prediction += 1

    class_name = classes[classes['Class_Number'] == prediction]['Class_Type'].values[0]
    
    return class_name

def run():
    dataset = pd.read_csv("zoo.csv")
    dataset['class_type'] = dataset['class_type'].astype('category')

    dataset = dataset.sample(frac=1).reset_index(drop=True)

    test_dataset = dataset.iloc[:10].reset_index(drop=True)

    dataset = dataset.drop(columns=['animal_name'])
    dataset = dataset.iloc[10:].reset_index(drop=True)


    X = dataset.drop(columns=['class_type']).values
    y = dataset['class_type'].cat.codes.values
    X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.10, random_state=1)

    model = DecisionTreeClassifier()
    model.fit(X_train, Y_train)

    ask_animal(test_dataset, model)

def ask_animal(dataset, model):
    root = tk.Tk()
    root.title("Select an Animal")

    # set the window to be in the center of the screen
    w = 250
    h = 150
    ws = root.winfo_screenwidth()
    hs = root.winfo_screenheight() 
    x = (ws/2) - (w/2)
    y = (hs/2) - (h/2)
    root.geometry('%dx%d+%d+%d' % (w, h, x, y))

    root.lift()
    root.focus_force()

    label = tk.Label(root, text="Select an animal from the list:")
    label.pack(pady=10)

    animal_names = dataset['animal_name'].tolist()
    variable = StringVar()
    
    w = OptionMenu(root, variable, *animal_names)
    w.pack()

    def select():
        selected_animal = variable.get()
        prediction = predict(selected_animal, model)
        
        root = tk.Tk()
        root.title("Prediction")
        w = 400
        h = 50
        ws = root.winfo_screenwidth()
        hs = root.winfo_screenheight() 
        x = (ws/2) - (w/2)
        y = (hs/2) - (h/2)
        root.geometry('%dx%d+%d+%d' % (w, h, x, y))

        root.lift()
        root.focus_force()

        label = tk.Label(root, text=f"The prediction for '{selected_animal}' is: {prediction}")
        label.pack(pady=10)

    button = ttk.Button(root, text="Get Prediction", command=select)
    button.pack(pady=10)

    root.mainloop()

if __name__ == "__main__":
    run()