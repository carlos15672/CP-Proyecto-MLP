import numpy as np
import time
from tensorflow.keras.datasets import fashion_mnist
#dataset

(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

# 28x28 a vectores 784
X_train = X_train.reshape(-1, 784) / 255.0
X_test = X_test.reshape(-1, 784) / 255.0

#config
entrada = 784
oculta = 128
salida = 10

learning_rate = 0.01
epocas = 5
batch_size = 128
#pesos
np.random.seed(42)

W1 = np.random.randn(entrada, oculta) * 0.01
b1 = np.zeros((1, oculta))

W2 = np.random.randn(oculta, salida) * 0.01
b2 = np.zeros((1, salida))
#function

def relu(x):
    return np.maximum(0, x)


def relu_derivada(x):
    return (x > 0).astype(float)


def softmax(x):
    x = x - np.max(x, axis=1, keepdims=True)
    exp = np.exp(x)
    return exp / np.sum(exp, axis=1, keepdims=True)
#entrenamiento
inicio = time.perf_counter()

for epoca in range(epocas):

    perdida_total = 0

    for i in range(0, len(X_train), batch_size):

        X = X_train[i:i + batch_size]
        y = y_train[i:i + batch_size]

        n = len(X)
#adelante

        z1 = X @ W1 + b1
        a1 = relu(z1)

        z2 = a1 @ W2 + b2
        pred = softmax(z2)
#loss

        loss = -np.mean(
            np.log(pred[np.arange(n), y] + 1e-8)
        )

        perdida_total += loss
#back
        dz2 = pred.copy()
        dz2[np.arange(n), y] -= 1
        dz2 /= n

        dW2 = a1.T @ dz2
        db2 = np.sum(dz2, axis=0, keepdims=True)

        da1 = dz2 @ W2.T

        dz1 = da1 * relu_derivada(z1)

        dW1 = X.T @ dz1
        db1 = np.sum(dz1, axis=0, keepdims=True)
#actualizate
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1

        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2


    print(
        f"Epoca {epoca + 1} | "
        f"Loss: {perdida_total:.4f}"
    )


fin = time.perf_counter()
#eval

z1 = X_test @ W1 + b1
a1 = relu(z1)

z2 = a1 @ W2 + b2
pred = softmax(z2)

clases = np.argmax(pred, axis=1)

accuracy = np.mean(clases == y_test) * 100


print("\nTiempo:", round(fin - inicio, 2), "segundos")
print("Accuracy:", round(accuracy, 2), "%")