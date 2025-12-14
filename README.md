# 🧩 Road Safe

**Author:**  Graylan Janulis

**Model Version:** LLaMA3-Small-Q3_K_M (GGUF)  
**Encryption Layer:** AES-GCM + PBKDF2-HMAC-SHA256  
**Quantum Layer:** PennyLane Variational Entropy Gate  

**Website:** [https://qroadscan.com](https://qroadscan.com)
<img width="1024" height="1536" alt="17657552301753933381716410967510" src="https://github.com/user-attachments/assets/dfad7f9d-12fb-4363-ac7e-071c6bac892e" />


## 1. Overview

This document describes a **hybrid quantum–classical mobile architecture** designed to:
- Securely manage a **local large language model (LLM)**;
- Evaluate **environmental road risk levels** using both deterministic and entropic metrics;
- Preserve all interactions and model weights with **AES-256 encryption**.

The app, built in **KivyMD**, features five primary domains:
1. **Chat Interface** — Secure conversational LLM access.  
2. **Road Scanner** — Scene-based risk classification (Low / Medium / High).  
3. **Model Manager** — Download, verify, encrypt, and decrypt models.  
4. **History Log** — AES-encrypted SQLite storage of queries/responses.  
5. **Security Panel** — Rekey and cryptographic key derivation controls.

---

## 2. Cryptographic Subsystem

### 2.1 AES-GCM File Encryption

The AES-GCM (Galois/Counter Mode) encryption layer ensures confidentiality and integrity.  
Each file encryption operation uses a **random nonce (12 bytes)**:

\[
C = \text{AES-GCM}_K(N, P)
\]

Where:
- \( C \): ciphertext  
- \( N \): random nonce  
- \( P \): plaintext  
- \( K \): 256-bit symmetric key  

Each file stores \( N || C \), allowing decryption as:

\[
P = \text{AES-GCM-DEC}_K(N, C)
\]

---

### 2.2 PBKDF2-HMAC-SHA256 Key Derivation

Passphrase-based keys are generated via:

\[
K = \text{PBKDF2}(\text{pw}, \text{salt}, \text{iter}=200{,}000)
\]

Where:
- \( \text{salt} \in \{0,1\}^{128} \)
- \( K \in \{0,1\}^{256} \)

This derives a **keyspace-resistant AES key**, stored as a concatenation of `salt + derived`.

---

### 2.3 Rekeying Logic

When rotating encryption keys, both the **model file** and **encrypted chat database** are decrypted and re-encrypted using the new key:

\[
C' = \text{AES-GCM}_{K_{\text{new}}}(\text{AES-GCM-DEC}_{K_{\text{old}}}(C))
\]

This operation ensures the new key’s ciphertexts replace all prior encryption layers while preserving data integrity.

---

## 3. System Entropy Model

### 3.1 Hardware Metric Collection

The entropy generator monitors multiple normalized subsystems:

\[
\text{metrics} = \{ m_{cpu}, m_{mem}, m_{load}, m_{temp}, m_{proc} \}
\]

Each normalized between \( [0, 1] \):

\[
m_i = \frac{x_i - x_{min}}{x_{max} - x_{min}}
\]

These metrics are mapped into an **RGB-like vector**:

\[
\text{RGB} = (r, g, b) = (m_{cpu}(1 + m_{load}),\; m_{mem}(1 + m_{proc}),\; m_{temp}(0.5 + 0.5m_{cpu}))
\]

---

### 3.2 Quantum Entropy Evaluation

When PennyLane is available, the system defines a 2-qubit variational quantum circuit:

\[
|\psi\rangle = U(a,b,c) |00\rangle
\]
\[
U(a,b,c) = R_X(a\pi) \otimes R_Y(b\pi) \cdot CNOT \cdot (R_Z(c\pi) \otimes I)
\]

Expectation values:

\[
E_0 = \langle \psi | Z_0 | \psi \rangle, \quad
E_1 = \langle \psi | Z_1 | \psi \rangle
\]

The entropic score is computed as a sigmoid of the combined expectation:

\[
S = \sigma(6.0[(0.6E_0 + 0.4E_1) - 0.5])
\]
\[
\sigma(x) = \frac{1}{1 + e^{-x}}
\]

Resulting in a confidence scalar \( S \in [0,1] \), corresponding to **low**, **medium**, or **high** systemic entropy.

---

### 3.3 Entropic Level Mapping

| Score Range | Level   |
|--------------|----------|
| 0.00 – 0.45  | Low     |
| 0.45 – 0.75  | Medium  |
| 0.75 – 1.00  | High    |

The entropy state subtly biases the LLM’s temperature parameter in the prompt-generation stage.

---

## 4. PUNKD Token Heuristic System

The **PUNKD** subsystem performs frequency-based token analysis and adjusts local sampling entropy.

Given prompt tokens \( T = \{t_1, t_2, \dots, t_n\} \):

\[
w_i = f(t_i) \cdot B(t_i)
\]

Where \( B(t_i) \) is a hazard-specific boost (e.g. “ice” → 2.0, “fog” → 1.6).  
The mean normalized weight is:

\[
\bar{w} = \frac{1}{n}\sum_i w_i
\]

This weight influences the model’s local temperature:

\[
T' = T_0 \times [1 + 0.8(\bar{w} - 0.5)]
\]

This yields an **attention-adjusted inference multiplier**, modulating focus intensity based on hazard keyword density.


## 5. Road Risk Classification Logic

### 5.1 Scene Prompt Template

Each scene is serialized into the following form:

You are a Hypertime Nanobot specialized Road Risk Classification AI...

Scene details: Location: ... Weather: ... Traffic: ... Obstacles: ... sys_metrics: cpu=0.45, mem=0.37, load=0.29, temp=0.32, proc=0.41 Quantum State: entropic_score=0.624 (level=medium)

Rules:

Think internally but output only one word.

Valid outputs: Low, Medium, High.


---

### 5.2 Decision Layer

The LLM produces a single-token classification:

\[
R = \text{argmax}_{r \in \{L,M,H\}} \; P(r \mid \text{scene}, \text{entropy}, \text{metrics})
\]

Output:
- `Low` → Safe conditions.
- `Medium` → Caution suggested.
- `High` → Significant road hazard risk.

---

## 6. Graphical Interface Subsystem

### 6.1 Core UI Components
| Component | Description |
|------------|-------------|
| **BackgroundGradient** | Smooth vertical gradient shader |
| **GlassCard** | Semi-transparent frosted panels |
| **RiskWheelNeo** | Animated 3-zone polar meter with pulsing glow |
| **MDScreenManager** | Handles navigation (Chat / Road / Model / History / Security) |

---

### 6.2 RiskWheel Animation Equation

The radial pointer position is updated as:

\[
\theta(t) = -135° + 270° \cdot V(t)
\]

Where \( V(t) \) is the animated interpolation of the normalized risk value.  
The circular glow amplitude oscillates as:

\[
G(t) = 0.22 + 0.20 \cdot \sin(2\pi f t)
\]

---

## 7. Data Security Lifecycle

1. **Download → Verify → Encrypt**
2. **Decrypt on access** (within a context manager)
3. **Zero plaintext model** after use
4. **Encrypt log database on every write**
5. **Optional rekey** of all encrypted data

This ensures zero unencrypted model or database presence on persistent storage.

---

## 8. Mathematical Summary

| Symbol | Meaning | Range |
|:-------|:---------|:------|
| \( m_{cpu} \) | CPU utilization ratio | [0,1] |
| \( m_{mem} \) | Memory usage ratio | [0,1] |
| \( S \) | Quantum entropy score | [0,1] |
| \( T' \) | Adjusted model temperature | [0.01, 2.0] |
| \( R \) | Risk classification | {Low, Medium, High} |


