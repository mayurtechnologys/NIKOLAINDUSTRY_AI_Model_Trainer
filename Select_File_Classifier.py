import tkinter as tk
from tkinter import ttk  # Add this line

from tkinter import filedialog, messagebox
from functools import partial
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import os

class TrainingProgressCallback(tf.keras.callbacks.Callback):
    def __init__(self, app, epochs):
        super().__init__()
        self.app = app
        self.epochs = epochs

    def on_epoch_end(self, epoch, logs=None):
        # Update the progress bar per epoch
        self.app.update_progress_bar((epoch + 1) / self.epochs)

class ModelTrainerApp:
    def __init__(self, master):
        self.master = master
        master.title("NIKOLAINDUSTRY AI Model Trainer")

        self.train_dir = ""
        self.test_dir = ""

        self.setup_ui()

    
    def setup_ui(self):
        # Using grid layout for better alignment
        self.master.columnconfigure(1, weight=1)

        # Training directory UI components
        self.train_label = tk.Label(self.master, text="Select training folder:")
        self.train_label.grid(row=0, column=0, sticky=tk.W)

        self.train_path_label = tk.Label(self.master, text="No folder selected", fg="grey")
        self.train_path_label.grid(row=0, column=1, sticky=tk.EW)

        self.train_button = tk.Button(self.master, text="Browse", command=self.browse_train)
        self.train_button.grid(row=0, column=2)

        # Testing directory UI components
        self.test_label = tk.Label(self.master, text="Select testing folder:")
        self.test_label.grid(row=1, column=0, sticky=tk.W)

        self.test_path_label = tk.Label(self.master, text="No folder selected", fg="grey")
        self.test_path_label.grid(row=1, column=1, sticky=tk.EW)

        self.test_button = tk.Button(self.master, text="Browse", command=self.browse_test)
        self.test_button.grid(row=1, column=2)

        # Epochs input
        self.epochs_label = tk.Label(self.master, text="Epochs:")
        self.epochs_label.grid(row=2, column=0, sticky=tk.W)

        self.epochs_entry = tk.Entry(self.master)
        self.epochs_entry.grid(row=2, column=1, sticky=tk.EW)
        self.epochs_entry.insert(0, "10")  # Default value

        # Batch size input
        self.batch_size_label = tk.Label(self.master, text="Batch Size:")
        self.batch_size_label.grid(row=3, column=0, sticky=tk.W)

        self.batch_size_entry = tk.Entry(self.master)
        self.batch_size_entry.grid(row=3, column=1, sticky=tk.EW)
        self.batch_size_entry.insert(0, "32")  # Default value

        # Training button
        self.train_button = tk.Button(self.master, text="Start Training", command=self.start_training)
        self.train_button.grid(row=4, column=0, columnspan=3)

        # Progress bar
        self.progress = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(self.master, variable=self.progress, maximum=100)  # Corrected line
        self.progress_bar.grid(row=5, column=0, columnspan=3, sticky=tk.EW)

    def browse_train(self):
        self.train_dir = filedialog.askdirectory()
        self.train_path_label.config(text=self.train_dir if self.train_dir else "No folder selected")

    def browse_test(self):
        self.test_dir = filedialog.askdirectory()
        self.test_path_label.config(text=self.test_dir if self.test_dir else "No folder selected")

    def start_training(self):
        if not self.train_dir or not self.test_dir:
            messagebox.showerror("Error", "Please select training and testing directories.")
            return

        epochs = int(self.epochs_entry.get())
        batch_size = int(self.batch_size_entry.get())

        train_datagen = ImageDataGenerator(rescale=1./255)
        test_datagen = ImageDataGenerator(rescale=1./255)

        train_generator = train_datagen.flow_from_directory(
            self.train_dir, target_size=(32, 32), batch_size=batch_size, class_mode='categorical')

        test_generator = test_datagen.flow_from_directory(
            self.test_dir, target_size=(32, 32), batch_size=batch_size, class_mode='categorical')

        num_classes = train_generator.num_classes

        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dense(num_classes, activation='softmax')
        ])

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        callbacks = [TrainingProgressCallback(self, epochs)]

        model.fit(train_generator, epochs=epochs, validation_data=test_generator, callbacks=callbacks)

        model.save('NIKOLAINDUSTRY_model_trained.h5')

        labels = (train_generator.class_indices)
        labels = dict((v,k) for k,v in labels.items())
        with open('labels.txt', 'w') as label_file:
            for key, value in labels.items():
                label_file.write(f"{key} {value}\n")

        messagebox.showinfo("Training Complete", "Model training has completed successfully. Model and labels saved.")

    def update_progress_bar(self, progress):
        self.progress.set(progress * 100)

if __name__ == "__main__":
    root = tk.Tk()
    app = ModelTrainerApp(root)
    root.mainloop()
