import numpy as np
import qutip as qt  # Quantum toolkit (stand-in for Qiskit)

# Step 1: Generate a space-traveling radio wave signal (~100 MHz, EM wave at light speed)
frequency = 100e6  # Hz, radio range (travels through space, invisible)
sample_rate = 1e9  # High sim rate for brevity
duration = 0.000001  # Short burst (1 us)
t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
wave = 0.5 * np.sin(2 * np.pi * frequency * t)  # Simple sine wave

# Step 2: "Reverse" the signal - phase invert
reverse_wave = -wave  # Inverts phase, like a 'negative' signal for cancellation effects

# Step 3: Add quantum superposition with QuTiP (like Qiskit) - make it "in and out" at once
# Model a qubit in superposition: |psi> = (|0> + |1>) / sqrt(2)  # "In" (1) and "out" (0) simultaneously
psi = (qt.basis(2, 0) + qt.basis(2, 1)).unit()  # Normalized superposition state
# Simulate measurement expectation for modulation (in reality, collapses on measure)
density_matrix = qt.ket2dm(psi)  # Density matrix for the state
modulation_op = qt.sigmax()  # Operator to get <X> expectation (measures superposition overlap)
modulation = qt.expect(modulation_op, density_matrix)  # Value ~1 for full superposition
quantum_wave = reverse_wave * (1 + modulation)  # Modulate wave to 'exist in both states'

# Step 4: Simulate propagation through space at light speed
c = 3e8  # m/s
distance = 1e6  # 1000 km through space
delay = distance / c  # Travel time
print(f"Simulated radio signal in superposition travels {distance} m in {delay:.2e} s at light speed.")

# Output sample data (fictional 'heard' via receiver demodulation)
print("Quantum-superposed wave sample:", quantum_wave[:10])
print("In simulation: Signal is 'in' (transmitted) and 'out' (not) simultaneously until 'measured'.")
