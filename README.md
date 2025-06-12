# Implementation of Neural Network from Scratch (in JavaScript)

A fully functional deep neural network built from scratch using **vanilla JavaScript** no frameworks, no libraries. Just pure math, logic, and a lot of debugging.

This project includes:
* A modular neural network architecture
* Support for sigmoid and ReLU activations
* Backpropagation
* Stochastic and mini-batch gradient descent(only in single hidden layer architecture)
* Custom loss function support (MSE, Cross-Entropy)
* Training support for classic datasets like **XOR**, **Iris**, and **MNIST**

---

## 🚀 Overview

This project was created to learn how neural networks work internally: weight updates, activation functions, and most importantly backpropagation.
Starting with a simple XOR network, I scaled it to the Iris dataset and eventually to MNIST, all while building layer-by-layer abstractions and a custom training loop.

Live Demo: [here](https://mohd-shamoon-04.github.io/NeuralNet/)

---

## 🧪 Technical Highlights

* **Initial XOR Network**
  Input: 2 neurons → Hidden: 2 neurons → Output: 1 neuron
  Proved why XOR is a classic test case (non-linearly separable).

* **Iris Dataset**
  Used a single hidden layer with 6 neurons and mini-batch gradient descent. Achieved **97% accuracy**.
  Then re-trained using a **deep NN** (two hidden layers: 6 and 4 neurons) and maintained the same accuracy.

* **Deep NN for MNIST**
  After refactoring layers into reusable components and adding support for softmax/cross-entropy, I trained on MNIST:

  ```
  Input: 784 neurons
  Hidden Layers: [18 (ReLU), 16 (ReLU)]
  Output: 10 neurons (Softmax)
  Loss: Cross-Entropy
  Accuracy: 96.5%
  ```

---

## 🧩 Problems Faced & Learnings

1. **Backpropagation**: Understanding and debugging gradients manually was the steepest learning curve.
2. **Hyperparameter Tuning**: Especially tricky with Iris, needed multiple learning rate and neuron count tweaks.
3. **Activation/Loss Mismatch**: Initially used sigmoid + MSE for MNIST. Accuracy jumped only after switching to ReLU (hidden) + Softmax (output) + Cross-Entropy loss.
4. **Modularization**: Refactored layers as independent objects to support flexible architectures.

---

## 📚 Resources

If you're looking to build something similar, here are some resources I found super helpful:

1. Videos by 3Blue1Brown - [*Neural Networks series*](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) 
2. Book by Michael Nielsen - [*Neural Networks and Deep Learning*](http://neuralnetworksanddeeplearning.com/)
3. Video by Mikael Laine - [*The Absolutely Simplest Neural Network Backpropagation Example*](https://www.youtube.com/watch?v=8d6jf7s6_Qs)


---

## 🛠️ How to Use

1. **Understand the Project Structure**
   Before running anything, browse the files to understand how the code is modularized (see [📁 File Structure](#-file-structure)).

2. **Run the MNIST Trained Model**
   * Navigate to the `mnist/` folder.
   * Run the following command:
     ```bash
     node model.js
     ```
   * ⚠️ Make sure you have **Node.js** installed.

3. **Run the Iris Trained Model**
   * Navigate to the `iris/` folder.  
   * Run:
     ```bash
     node model.js
     ```

4. **Custom Training**
   * To experiment with your own datasets or tweak hyperparameters, modify the code accordingly.
   * You can adjust learning rate, epoch count, architecture, etc., and retrain from scratch.

---

## 📁 File Structure

```
| neuralNet/
|
|-- JS/  
|--|-- VNN.js               # neural network implementation  
|--|-- test1.js             # tests  
|--|-- test2.js             # tests  
|--|-- OGxorNN.js           # first fully function neural network(xor)  
|
|-- iris/  
|--|-- model.js             # model train and evaluate  
|--|-- load.js              # helper module to load iris dataset  
|--|-- model.json           # trained model weights and biases  
|
|-- mnist/  
|--|-- model.js             # model train and evaluate  
|--|-- load.js              # helper module to load mnist dataset  
|--|-- webModel.json        # updated VNN.js file for in browser execution of model  
|--|-- dump/                
|--|--|-- WebModelNew.json  # trained model weights and biases  
|
|-- index.html              # entry point for GitHub pages(live demo)  
|
|-- visualize/  
|--|-- webModel.json        # updated VNN.js file for direct browser execution of model  
|--|-- handwritten.html     # demo for model trained on mnist dataset  
|--|-- iris.html            # demo for model trained on iris dataset  
```