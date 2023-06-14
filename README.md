# Implementation of algarithms A3C and MBVE in MuJoCo in Python
## A3C algorithm
```
// Assume global shared parameter vectors θ and θv and global shared counter T = 0 
// Assume thread-specific parameter vectors θ0 and θ0v
Initialize thread step counter t ← 1
repeat
Reset gradients: dθ ← 0 and dθv ← 0.
    Synchronize thread-specific parameters θ0 = θ and θ0v = θv
    tstart = t
    Get state st
    repeat
        Perform at according to policy π(at|st; θ0)
        Receive reward rt and new state st+1
        t ← t + 1
        T ← T + 1
    until terminal st or t − tstart == tmax
    R = 0 for terminal st
    or
    R = V(st, θ0v) for non - terminal st // Bootstrap from last state
    for i ∈ {t − 1, . . . , tstart} do
        R ← ri + γR
        Accumulate gradients wrt θ0: dθ ← dθ + ∇θ0 log π(ai|si; θ0)(R − V (si; θ0v))
        Accumulate gradients wrt θ0v: dθv ← dθv + ∂ (R − V (si; θ0v))2/∂θ0v
    end for
    Perform asynchronous update of θ using dθ and of θv using dθv.
until T > Tmax
```
## MBVE algorithm
MBVE is based on DDGP with modification of including information about future rewards and states based of the model of the environment.
In our case we use second mujoco simulation as a perfect model. Most important part of the algorithm is located in learn() method of Agent class.
