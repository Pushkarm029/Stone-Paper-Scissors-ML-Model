## **Stone🪨 Paper📄 Scissors✂️**

### A CNN Model that can classify Stone Paper Scissors made using Tensorflow and Keras.
### Accuracy of 86% on the validation set.
#### ***⚠️Disclaimer***: The model is not perfect and gives max accuracy when the object is placed in the center of the frame and fits the frame.

### Dataset: [Stone Paper Scissors Dataset](https://universe.roboflow.com/roboflow-58fyf/rock-paper-scissors-sxsw)

### Modes of Use:
1. **Image**
2. **Video**
3. **Webcam**

### Model Architecture:
```python
   cnn = models.Sequential([
       layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(32,32, 3)),
       layers.MaxPooling2D((2, 2)),
       layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
       layers.MaxPooling2D((2, 2)),

       layers.Flatten(),
       layers.Dense(64, activation='relu'),
       layers.Dense(10, activation='softmax')
   ])
```

```python
   cnn.compile(optimizer='adam',
                   loss='sparse_categorical_crossentropy',
                   metrics=['accuracy'])
```

### Precautions:
- **In Webcam mode or Video Mode**, If you see too many wrong predictions, try to reduce frame rate in repective code (for video -> video.py, for webcam -> camera.py).