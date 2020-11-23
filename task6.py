import tkinter as tk
import subprocess
from tkinter import ttk
from tkinter import scrolledtext


#submit button function
def submit():
    #parameters
    query_text=query_entry.get()
    param_text=param_entry.get()
    task_selector = feedback_select.get()

    #Clear before new submission
    results.delete(0.0, tk.END)
    task4arg = "TODO"
    task5arg = "TODO"

    #radiobutton handler
    if task_selector == 'task4':
        results.insert(tk.END, 'PR feedback TODO')
        #subprocess.call(['python', 'task4.py', task4arg])
    elif task_selector == 'task5':
        results.insert(tk.END, 'CB feedback TODO')
        #subprocess.call(['python', 'task5.py', task5arg])
    else:
        results.insert(tk.END, 'No feedback TODO')
        #TOFIX: Unsure if this is necessary: depends on T4/T5

#Window Settings
window = tk.Tk()
window.title("Task 6")
window.geometry("800x500")
ttk.Label (window, text="Query and Feedback Interface\n",font="none 16").grid(row=0,column=0, columnspan=2,sticky=tk.W)

#Provide query and parameters
ttk.Label (window,text="Enter a query: [FORMAT TODO: gesture file]",font="none 12").grid(row=1,column=0,sticky=tk.W)
query_entry = tk.Entry(window,width=40)
query_entry.grid(row=1,column=1,sticky=tk.W)

ttk.Label (window, text="Enter number of results: [FORMAT TODO]",font="none 12").grid(row=2,column=0,sticky=tk.W)
param_entry = tk.Entry(window,width=40)
param_entry.grid(row=2,column=1,sticky=tk.W)

ttk.Label (window, text="Enter relevance parameters: [FORMAT TODO]",font="none 12").grid(row=4,column=0,sticky=tk.W)
param_entry = tk.Entry(window,width=40)
param_entry.grid(row=5,column=0,sticky=tk.W)

#Task4 or Task5
feedback_select = tk.StringVar()
task4_select = ttk.Radiobutton(window, text="Probabilistic Relevance", variable=feedback_select, value='task4')
task4_select.grid(row=4, column=1,sticky=tk.W)
task5_select = ttk.Radiobutton(window, text="Classifier-based Relevance", variable=feedback_select, value='task5')
task5_select.grid(row=5, column=1,sticky=tk.W)

#submit button
tk.Button(window,text="SUBMIT QUERY",width=20,command=submit).grid(row=6,column=0,sticky=tk.W)

#Show results
results = scrolledtext.ScrolledText(window, width=60, height=10,wrap=tk.WORD)
results.grid(row=7,column=0,columnspan=2,sticky=tk.W)

####Run GUI####
window.mainloop()