# 🧠 Machine-Learning-libraries-in-C

A lightweight, dependency-free tensor math and neural network library written in pure C. This library provides a memory-safe `Tensor` struct and core operations including matrix multiplication, addition, and activation functions like ReLU and Sigmoid. 

It is designed for low-level machine learning projects where you need absolute control over memory and mathematical operations without heavy external dependencies like PyTorch or TensorFlow.

---

## 🏗️ Computational Flow

The library computes forward passes through basic tensor operations. Here is the lifecycle of a typical neural network layer in `C-ML-Library`:

```mermaid
flowchart LR
    A[Initialize Tensors] --> B[Populate Weights & Inputs]
    B --> C[Matrix Multiplication <br> <code>tensor_matmul()</code>]
    C --> D[Apply Activation <br> <code>tensor_relu()</code>]
    D --> E[Output Result Tensor]
    
    style A fill:#eaddc9,stroke:#4c3d32
    style B fill:#eaddc9,stroke:#4c3d32
    style C fill:#f4ebdc,stroke:#4c3d32
    style D fill:#f4ebdc,stroke:#4c3d32
    style E fill:#d8cbb8,stroke:#4c3d32,stroke-width:2px
```

---

## 🕸️ Neural Network Architecture Example

Using the provided operations, you can easily build Multi-Layer Perceptrons (MLPs). The example provided in `src/main.c` executes a forward pass of a simple layer.

```mermaid
graph TD
    subgraph Input Layer (3x1 Tensor)
        I1((x1 = 2.0))
        I2((x2 = 2.0))
        I3((x3 = 2.0))
    end

    subgraph Output Layer (2x1 Tensor)
        H1((y1 = 3.0))
        H2((y2 = 3.0))
    end

    I1 -- "w = 0.5" --> H1
    I1 -- "w = 0.5" --> H2
    I2 -- "w = 0.5" --> H1
    I2 -- "w = 0.5" --> H2
    I3 -- "w = 0.5" --> H1
    I3 -- "w = 0.5" --> H2
    
    classDef layer fill:#eaddc9,stroke:#4c3d32,stroke-width:1px;
    class I1,I2,I3,H1,H2 layer;
```

---

## 🚀 Running and Testing

To compile and run the provided test in `src/main.c`, use standard `gcc` or `clang` and remember to link the math library (`-lm`).

### Compilation
```bash
gcc src/main.c src/ml.c -Iinclude -lm -o ml_app
```

### Execution
```bash
./ml_app
```

### Expected Test Output
When you run the test, the library initializes the weights to `0.5`, inputs to `2.0`, performs a matrix multiplication, and passes the result through a ReLU activation. The resulting graph perfectly calculates the values:

```text
c-ml-library: Initializing Neural Network Layers...

Result of Forward Pass (MatMul + ReLU):
Tensor(2, 1):
3.0000 
3.0000 
```

---

## 📚 Documentation

The repository includes a beautifully styled, fully responsive documentation website located in the `docs/` folder. 

Open `docs/index.html` in your web browser to explore:
- Detailed API Reference (`tensor_create`, `tensor_free`, `tensor_fill`, `tensor_print`)
- Math Operations & Activations
- Full Integration Examples

---

**License:** MIT
**Author:** Ahmad Shajhan
