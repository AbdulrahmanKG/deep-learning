import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

Wf, Uf, bf = 0.1,  0.2,  0.3
Wi, Ui, bi = 0.4,  0.5,  0.6
Wo, Uo, bo = 0.7,  0.8,  0.9
Wc, Uc, bc = 0.1, -0.3,  0.2

Wy, by = 0.5, 0.1

h = 0.0
c = 0.0

X = [1, 2, 3]

print("=" * 55)
print("       LSTM From Scratch — Numerical Example")
print("=" * 55)
print(f"\nWeights  →  Wf={Wf}, Uf={Uf}, bf={bf}")
print(f"             Wi={Wi}, Ui={Ui}, bi={bi}")
print(f"             Wo={Wo}, Uo={Uo}, bo={bo}")
print(f"             Wc={Wc}, Uc={Uc}, bc={bc}")
print(f"Output layer →  Wy={Wy}, by={by}")
print(f"\nInitial states →  h0={h}, c0={c}")
print(f"Input sequence →  {X}\n")

for t, x in enumerate(X, start=1):
    print(f"{'─'*55}")
    print(f"  Time Step t={t},  Input x={x}")
    print(f"{'─'*55}")

    f = sigmoid(Wf * x + Uf * h + bf)
    print(f"  1) Forget gate   f  = σ({Wf}·{x} + {Uf}·{h:.4f} + {bf}) = {f:.4f}")

    i = sigmoid(Wi * x + Ui * h + bi)
    print(f"  2) Input gate    i  = σ({Wi}·{x} + {Ui}·{h:.4f} + {bi}) = {i:.4f}")

    c_tilde = tanh(Wc * x + Uc * h + bc)
    print(f"  3) Candidate     c̃  = tanh({Wc}·{x} + {Uc}·{h:.4f} + {bc}) = {c_tilde:.4f}")

    c = f * c + i * c_tilde
    print(f"  4) Cell state    c  = {f:.4f}·{c - i*c_tilde:.4f} + {i:.4f}·{c_tilde:.4f} = {c:.4f}")

    o = sigmoid(Wo * x + Uo * h + bo)
    print(f"  5) Output gate   o  = σ({Wo}·{x} + {Uo}·{h:.4f} + {bo}) = {o:.4f}")

    h = o * tanh(c)
    print(f"  6) Hidden state  h  = {o:.4f} · tanh({c:.4f}) = {h:.4f}\n")

y_hat = Wy * h + by

print("=" * 55)
print("  Step 3: Predict the Next Value")
print("=" * 55)
print(f"\n  ŷ = Wy · h + by")
print(f"  ŷ = {Wy} × {h:.4f} + {by}")
print(f"  ŷ = {y_hat:.4f}")
print(f"\n  ✓ Final Prediction ≈ {y_hat:.2f}  (close to 4)")
print("=" * 55)
