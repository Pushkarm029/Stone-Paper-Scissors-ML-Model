## **Stone🪨 Paper📄 Scissors✂️**

### Stone Paper Scissors is a CNN Model built with Tensorflow and Keras that can classify hand gestures into three categories.
### Accuracy of **86%** on the validation set.
#### ***⚠️Disclaimer***: The model is not perfect and gives max accuracy when the object is placed in the center of the frame and fits the frame.
#### **Web Version** will be available soon.
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

### Other Things that i tried but didn't work🥲 :

1. **Intersection Over Union (IOU)** :With this, the model can detect and classify hand gestures even if the object is not in the centre of the frame. You can see the code for it in the `Other_Exps/test.py` file. The problem I got is that I somehow managed to get the required boundaries of the object, which are already preset in datasets. I was confused about how to get boundaries for new external images. So, I dropped this idea.
2. **Cascade Classifier** : I tried to use the Cascade Classifier to detect hand gestures and then classify them, but it didn't work well because its accuracy is very low and it only managed to detect *STONE*. You can see the code for it in the `Other_Exps/camera_live.py` file.
