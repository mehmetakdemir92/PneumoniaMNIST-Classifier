from medmnist import PneumoniaMNIST
from medmnist import INFO
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve


info = INFO['pneumoniamnist']
DataClass = PneumoniaMNIST

data_transform = transforms.Compose([transforms.ToTensor()])
train_dataset = DataClass(split='train', transform=data_transform, download=True)
val_dataset = DataClass(split='val', transform=data_transform, download=True)
test_dataset = DataClass(split='test', transform=data_transform, download=True)

train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

labels = train_dataset.labels.ravel()
class_counts = Counter(labels)
print("Train Set Distribution:", class_counts)

def show_samples(dataset, title):
    fig, axes = plt.subplots(1, 5, figsize=(12, 3))
    fig.suptitle(title)
    class0 = [img[0] for img in dataset if img[1] == 0][:5]
    class1 = [img[0] for img in dataset if img[1] == 1][:5]

    for i in range(5):
        axes[i].imshow(class0[i].squeeze(), cmap='gray')
        axes[i].set_title("Normal")
        axes[i].axis('off')

    plt.show()

    fig, axes = plt.subplots(1, 5, figsize=(12, 3))
    for i in range(5):
        axes[i].imshow(class1[i].squeeze(), cmap='gray')
        axes[i].set_title("Pneumonia")
        axes[i].axis('off')

    plt.show()

show_samples(train_dataset, "Sample Images")

def plot_pixel_distribution(dataset, title):
    class0_pixels = []
    class1_pixels = []

    for img, label in dataset:
        pixels = img.numpy().flatten()
        if label == 0:
            class0_pixels.extend(pixels)
        else:
            class1_pixels.extend(pixels)

    plt.figure(figsize=(10, 4))
    plt.hist(class0_pixels, bins=50, alpha=0.5, label='Normal', color='blue')
    plt.hist(class1_pixels, bins=50, alpha=0.5, label='Pneumonia', color='red')
    plt.title(f'Pixel Intensity Distribution - {title}')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

plot_pixel_distribution(train_dataset, "Train Set")


def compute_class_weights(counter):
    total = sum(counter.values())
    return {cls: total/count for cls, count in counter.items()}

class_weights = compute_class_weights(class_counts)
print("Class Weights:", class_weights)

def get_kernel(kernel_name):
    if kernel_name == 'sobel':
        return np.array([[1, 0, -1],
                         [2, 0, -2],
                         [1, 0, -1]])
    elif kernel_name == 'emboss':
        return np.array([[-2, -1,  0],
                         [-1,  1,  1],
                         [ 0,  1,  2]])
    else:
        raise ValueError("Unsupported kernel.")


def conv2d(X, kernel):
    kh, kw = kernel.shape
    h, w = X.shape
    pad = kh // 2
    padded = np.pad(X, pad, mode='constant')
    result = np.zeros_like(X)

    for i in range(h):
        for j in range(w):
            region = padded[i:i + kh, j:j + kw]
            result[i, j] = np.sum(region * kernel)

    return result


def max_pooling(X, pool_size=(2, 2)):
    ph, pw = pool_size
    h, w = X.shape
    pooled = np.zeros((h // ph, w // pw))

    for i in range(0, h, ph):
        for j in range(0, w, pw):
            region = X[i:i + ph, j:j + pw]
            pooled[i // ph, j // pw] = np.max(region)

    return pooled

def flatten(X):
    return X.flatten().reshape(-1, 1)

def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)

def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)

def initialize_dense_layer(input_dim, output_dim):
    weights = np.random.randn(output_dim, input_dim) * 0.01
    biases = np.zeros((output_dim, 1))
    return weights, biases


def dense_layer(X, weights, biases):
    return np.dot(weights, X) + biases


def output_layer(X, weights, biases):
    z = np.dot(weights, X) + biases
    return sigmoid(z)

def apply_activation(X, activation='relu'):
    if activation == 'relu':
        return relu(X)
    elif activation == 'sigmoid':
        return sigmoid(X)
    else:
        raise ValueError("Unsupported activation.")


def forward_propagation(X, parameters, kernels, activation1='relu', activation2='sigmoid', dropout_rate=0.5):
    cache = {}

    Z1 = conv2d(X, kernels[0])
    A1 = apply_activation(Z1, activation1)
    P1 = max_pooling(A1)
    cache['Z1'], cache['A1'], cache['P1'] = Z1, A1, P1

    Z2 = conv2d(P1, kernels[1])
    A2 = apply_activation(Z2, activation2)
    P2 = max_pooling(A2)
    cache['Z2'], cache['A2'], cache['P2'] = Z2, A2, P2

    F = flatten(P2)
    cache['F'] = F

    W_dense, b_dense = parameters['W_dense'], parameters['b_dense']
    Z3 = dense_layer(F, W_dense, b_dense)
    A3 = apply_activation(Z3, activation='relu')
    cache['Z3'], cache['A3'] = Z3, A3
    cache['W_dense'], cache['b_dense'] = W_dense, b_dense

    mask = (np.random.rand(*A3.shape) > dropout_rate).astype(float)
    A3 *= mask
    cache['dropout_mask'] = mask

    W_out, b_out = parameters['W_out'], parameters['b_out']
    output = output_layer(A3, W_out, b_out)
    cache['output'] = output
    cache['W_out'], cache['b_out'] = W_out, b_out

    return output, cache


def compute_loss(Y, Y_hat, loss_type='binary_cross_entropy'):
    eps = 1e-8
    Y_hat = np.clip(Y_hat, eps, 1 - eps)

    if loss_type == 'binary_cross_entropy':
        loss = - (Y * np.log(Y_hat) + (1 - Y) * np.log(1 - Y_hat))
        return np.squeeze(loss)
    else:
        raise ValueError("Unsupported loss type.")


def backprop_output_layer(Y, Y_hat, cache):
    A_prev = cache['A3']
    W_out = cache['W_out']

    Y = Y.reshape(Y.shape[0], -1)
    Y_hat = Y_hat.reshape(Y_hat.shape[0], -1)
    A_prev = A_prev.reshape(A_prev.shape[0], -1)

    m = Y.shape[1]

    dZ = Y_hat - Y

    dW_out = np.dot(dZ, A_prev.T) / m
    db_out = np.sum(dZ, axis=1, keepdims=True) / m
    dA_prev = np.dot(W_out.T, dZ)

    grads = {
        'dW_out': dW_out,
        'db_out': db_out,
        'dA3': dA_prev
    }

    return grads


def backprop_dense_layer(dA, cache, activation='relu'):
    Z = cache['Z3']
    A_prev = cache['F']
    W = cache['W_dense']

    Z = Z.reshape(Z.shape[0], -1)
    A_prev = A_prev.reshape(A_prev.shape[0], -1)
    dA = dA.reshape(dA.shape[0], -1)

    m = A_prev.shape[1]

    if activation == 'relu':
        dZ = dA * relu_derivative(Z)
    elif activation == 'sigmoid':
        dZ = dA * sigmoid_derivative(Z)
    else:
        raise ValueError("Unsupported activation function")

    dW = np.dot(dZ, A_prev.T) / m
    db = np.sum(dZ, axis=1, keepdims=True) / m
    dA_prev = np.dot(W.T, dZ)

    grads = {
        'dW_dense': dW,
        'db_dense': db,
        'dF': dA_prev
    }

    return grads


def update_parameters(parameters, grads, learning_rate=0.01):
    parameters['W_dense'] -= learning_rate * grads['dW_dense']
    parameters['b_dense'] -= learning_rate * grads['db_dense']

    parameters['W_out'] -= learning_rate * grads['dW_out']
    parameters['b_out'] -= learning_rate * grads['db_out']

    return parameters

def dataset_to_numpy(dataset):
    images = []
    labels = []
    for img, label in dataset:
        images.append(img.numpy().squeeze())
        labels.append(label)
    return np.array(images), np.array(labels)

X_train, Y_train = dataset_to_numpy(train_dataset)
X_val, Y_val = dataset_to_numpy(val_dataset)
X_test, Y_test = dataset_to_numpy(test_dataset)

kernels = [get_kernel("sobel"), get_kernel("emboss")]

dummy_input = X_train[0].squeeze()

Z1 = conv2d(dummy_input, kernels[0])
A1 = apply_activation(Z1, activation='relu')
P1 = max_pooling(A1)

Z2 = conv2d(P1, kernels[1])
A2 = apply_activation(Z2, activation='sigmoid')
P2 = max_pooling(A2)

F = flatten(P2)
flattened_dim = F.shape[0]
print("Detected flattened dim:", flattened_dim)

W_dense, b_dense = initialize_dense_layer(flattened_dim, 16)
W_out, b_out = initialize_dense_layer(16, 1)

parameters = {
    'W_dense': W_dense,
    'b_dense': b_dense,
    'W_out': W_out,
    'b_out': b_out
}


def train_model(X_train, Y_train, parameters, kernels, epochs=10, learning_rate=0.01, loss_type='binary_cross_entropy'):
    losses = []

    for epoch in range(epochs):
        total_loss = 0

        for i in range(X_train.shape[0]):
            x = X_train[i]
            y = Y_train[i]

            x = x.squeeze()
            y = np.array([[y]])

            y_hat, cache = forward_propagation(x, parameters, kernels)

            loss = compute_loss(y, y_hat, loss_type=loss_type)
            total_loss += loss

            grads_out = backprop_output_layer(y, y_hat, cache)
            grads_dense = backprop_dense_layer(grads_out['dA3'], cache)

            grads = {
                'dW_out': grads_out['dW_out'],
                'db_out': grads_out['db_out'],
                'dW_dense': grads_dense['dW_dense'],
                'db_dense': grads_dense['db_dense']
            }

            parameters = update_parameters(parameters, grads, learning_rate)

        avg_loss = total_loss / X_train.shape[0]
        losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

    return parameters, losses

trained_parameters, train_losses = train_model(X_train, Y_train, parameters, kernels, epochs=10)

def evaluate_model(X, Y, parameters, kernels, threshold=0.5, plot_roc=False, title="ROC Curve"):
    Y_preds = []
    Y_probs = []

    for i in range(X.shape[0]):
        x = X[i].squeeze()
        y_hat, _ = forward_propagation(x, parameters, kernels)
        prob = y_hat.item()
        Y_probs.append(prob)
        Y_preds.append(1 if prob >= threshold else 0)

    Y_true = Y.flatten()

    acc = accuracy_score(Y_true, Y_preds)
    prec = precision_score(Y_true, Y_preds)
    rec = recall_score(Y_true, Y_preds)
    f1 = f1_score(Y_true, Y_preds)
    cm = confusion_matrix(Y_true, Y_preds)
    auc = roc_auc_score(Y_true, Y_probs)

    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp)

    print("Accuracy:", acc)
    print("Precision:", prec)
    print("Recall:", rec)
    print("F1-Score:", f1)
    print("Specificity:", specificity)
    print("AUC:", auc)
    print("Confusion Matrix:\n", cm)

    if plot_roc:
        fpr, tpr, _ = roc_curve(Y_true, Y_probs)
        plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.show()

    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'specificity': specificity,
        'auc': auc,
        'confusion_matrix': cm
    }

val_accuracies = []
val_precisions = []
val_recalls = []
val_f1s = []

for epoch in range(10):
    print(f"\nMetrics for Epoch {epoch}:")
    metrics = evaluate_model(X_val, Y_val, trained_parameters, kernels)
    val_accuracies.append(metrics['accuracy'])
    val_precisions.append(metrics['precision'])
    val_recalls.append(metrics['recall'])
    val_f1s.append(metrics['f1'])

kernels = [get_kernel("sobel"), get_kernel("emboss")]

print("\nValidation Evaluation:")
evaluate_model(X_val, Y_val, trained_parameters, kernels, plot_roc=True ,title="ROC Curve - Validation")

print("\nTest Evaluation:")
evaluate_model(X_test, Y_test, trained_parameters, kernels, plot_roc=True ,title="ROC Curve - Test")

plt.plot(train_losses, label='Train Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Over Epochs")
plt.legend()
plt.grid(True)
plt.show()

plt.plot(val_accuracies, label='Accuracy')
plt.plot(val_precisions, label='Precision')
plt.plot(val_recalls, label='Recall')
plt.plot(val_f1s, label='F1-Score')
plt.xlabel("Epoch")
plt.ylabel("Score")
plt.title("Validation Metrics Over Epochs (Simulated)")
plt.legend()
plt.grid(True)
plt.show()