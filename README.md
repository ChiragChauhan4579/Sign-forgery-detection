
# Sign Forgery Detection

A deep learning based web app that compares the similarity between the two images of the signature and predicts if the signature matches or not. 

App Link: https://sign-forgery-detection.herokuapp.com/

Dataset Link: https://www.kaggle.com/robinreni/signature-verification-dataset

![demo](https://github.com/ChiragChauhan4579/Sign-forgery-detection/blob/main/1623143565581.gif)
## Run Locally

Clone the project

```bash
  git clone https://github.com/ChiragChauhan4579/Sign-forgery-detection
```

Go to the project directory

```bash
  cd Sign-forgery-detection
```

Install dependencies

```bash
  pip freeze > requirements.txt
```

Make sure to add your database details and change allowed host in settings.py file.

Start the server

```bash
  python manage.py runserver
```

  
## Documentation

After importing the dataset, the images are combined in pair of 2 as combination of either real-real or real-fake signatures. By using the for loop the images are resized and returned properly with the function.

```python
def read_data(dir, data):
    images1 = [] 
    images2 = [] 
    labels = []
    for j in range(0, len(data)):
        path = os.path.join(dir,data.iat[j, 0])
        img1 = cv2.imread(path)
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img1 = cv2.resize(img1, (100, 100))
        images1.append([img1])
        path = os.path.join(dir, data.iat[j, 1])
        img2 = cv2.imread(path)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        img2 = cv2.resize(img2, (100, 100))
        images2.append([img2])
        labels.append(np.array(data.iat[j, 2]))
    images1 = np.array(images1).astype(np.float32) / 255.0
    images2 = np.array(images2).astype(np.float32) / 255.0
    labels = np.array(labels).astype(np.float32)
    return images1, images2, labels

train_dir = '/content/sign_data/train'
train_csv = '/content/sign_data/train_data.csv'
df_train = pd.read_csv(train_csv, header=None)
train_images1, train_images2, train_labels = read_data(dir=train_dir, data=df_train)
train_labels = to_categorical(train_labels)

size = 100
train_images1 = train_images1.reshape(-1, size, size, 1)
train_images2 = train_images2.reshape(-1, size, size, 1)
```

A simple CNN base network and a function to calculate the euclidean distance between images and then a function on how would be the shape of that returned distance.


```python
def initialize_base_network(input_shape):
    clf = Sequential()
    clf.add(Convolution2D(64, (3,3),input_shape=input_shape))
    clf.add(Activation('relu'))
    clf.add(MaxPooling2D(pool_size=(2, 2)))
    clf.add(Convolution2D(32, (3,3)))
    clf.add(Activation('relu'))
    clf.add(MaxPooling2D(pool_size=(2, 2)))
    clf.add(Flatten())
    clf.add(Dense(128, activation='relu'))
    clf.add(Dense(64, activation='relu'))
    return clf
def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))
def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)
```

With the help of Model function of keras one can make the model as follow. The distance is calculated into the lambda layer.

```python
input_dim = (100, 100, 1)
base_network = initialize_base_network(input_dim)
img_a = Input(shape=input_dim)
img_b = Input(shape=input_dim)
vec_a = base_network(img_a)
vec_b = base_network(img_b)
distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([vec_a, vec_b])
prediction = Dense(2, activation='softmax')(distance)
model = Model([img_a, img_b], prediction)
model.summary()
```

Now the model can be trained as

```python
adam = tf.keras.optimizers.Adam(lr=0.00008)
model.compile(loss='categorical_crossentropy',optimizer=adam,metrics=['accuracy'])
model.fit([train_images1,train_images2],train_labels,validation_split=.30,batch_size=32,epochs=40)
```
## Optimizations

Trying different model architectures to get a more better working model.
  
