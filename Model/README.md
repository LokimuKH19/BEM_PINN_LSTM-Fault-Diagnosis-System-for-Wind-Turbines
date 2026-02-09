# 🔧 Technical Details of the PINN-LSTM Fault Diagnosis

> Detailed description corresponding to the extended abstract "ICOPE-2026-51023".

## 🎯 The Principal of Wind Turbine Fault Detection

For wind turbines, the rotor dynamics obeys the following angular momentum equation:

```math
  J\frac{d\Omega}{dT} = T_{aero}(V, \Omega, \theta) - T_{gen} - T_{in}
```

where $T_{aero}$ is the torque at the aero-dynamics side, determined by the wind speed $V$, rotor speed $\Omega$ and the pitch angle $\theta$, while $T_{gen}, T_{in}$ denotes the torque of generator, and the dynamic loss (elastic, damping term etc.) respectively. 

The wind turbines is controlled by adjusting the pitch angle $\theta$ to capture the wind energy at a certain wind speed , and the rotor speed $\Omega$ is determined according to the previous dynamic equation.

Therefore, if the system experiences mechanical failure which leads to deviation of the dynamic constants, such as the rotor inertia $J$, the rotor speed $\Omega$ would drift at the same wind speed $V$ and pitch angle $\theta$, and further influence the mechanical quantities. Consequently, our objective is to diagnose the fault from the Tower Thrust $F_t$ and the Torque at the aero-dynamic side $T_{aero}$ signals in real-time.

## ⚙ The Design of BEM-PINN

