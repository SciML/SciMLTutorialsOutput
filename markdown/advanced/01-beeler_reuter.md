---
author: "Shahriar Iravanian"
title: "An Implicit/Explicit CUDA-Accelerated Solver for the 2D Beeler-Reuter Model"
---


## Background

[SciML](https://github.com/SciML) is a suite of optimized Julia libraries to solve ordinary differential equations (ODE). *SciML* provides a large number of explicit and implicit solvers suited for different types of ODE problems. It is possible to reduce a system of partial differential equations into an ODE problem by employing the [method of lines (MOL)](https://en.wikipedia.org/wiki/Method_of_lines). The essence of MOL is to discretize the spatial derivatives (by finite difference, finite volume or finite element methods) into algebraic equations and to keep the time derivatives as is. The resulting differential equations are left with only one independent variable (time) and can be solved with an ODE solver. [Solving Systems of Stochastic PDEs and using GPUs in Julia](http://www.stochasticlifestyle.com/solving-systems-stochastic-pdes-using-gpus-julia/) is a brief introduction to MOL and using GPUs to accelerate PDE solving in *JuliaDiffEq*. Here we expand on this introduction by developing an implicit/explicit (IMEX) solver for a 2D cardiac electrophysiology model and show how to use [CUDA](https://github.com/JuliaGPU/CUDA.jl) libraries to run the explicit part of the model on a GPU.

Note that this tutorial does not use the [higher order IMEX methods built into DifferentialEquations.jl](https://docs.sciml.ai/latest/solvers/split_ode_solve/#Implicit-Explicit-(IMEX)-ODE-1) but instead shows how to hand-split an equation when the explicit portion has an analytical solution (or approxiate), which is common in many scenarios.

There are hundreds of ionic models that describe cardiac electrical activity in various degrees of detail. Most are based on the classic [Hodgkin-Huxley model](https://en.wikipedia.org/wiki/Hodgkin%E2%80%93Huxley_model) and define the time-evolution of different state variables in the form of nonlinear first-order ODEs. The state vector for these models includes the transmembrane potential, gating variables, and ionic concentrations. The coupling between cells is through the transmembrame potential only and is described as a reaction-diffusion equation, which is a parabolic PDE,

$$\partial V / \partial t = \nabla (D  \nabla V) - \frac {I_\text{ion}} {C_m},$$

where $V$ is the transmembrane potential, $D$ is a diffusion tensor, $I_\text{ion}$ is the sum of the transmembrane currents and is calculated from the ODEs, and $C_m$ is the membrane capacitance and is usually assumed to be constant. Here we model a uniform and isotropic medium. Therefore, the model can be simplified to,

$$\partial V / \partial t = D \Delta{V} - \frac {I_\text{ion}} {C_m},$$

where $D$ is now a scalar. By nature, these models have to deal with different time scales and are therefore classified as *stiff*. Commonly, they are solved using the explicit Euler method, usually with a closed form for the integration of the gating variables (the Rush-Larsen method, see below). We can also solve these problems using implicit or semi-implicit PDE solvers (e.g., the [Crank-Nicholson method](https://en.wikipedia.org/wiki/Crank%E2%80%93Nicolson_method) combined with an iterative solver). Higher order explicit methods such as Runge-Kutta and linear multi-step methods cannot overcome the stiffness and are not particularly helpful.

In this tutorial, we first develop a CPU-only IMEX solver and then show how to move the explicit part to a GPU.

### The Beeler-Reuter Model

We have chosen the [Beeler-Reuter ventricular ionic model](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1283659/) as our example. It is a classic model first described in 1977 and is used as a base for many other ionic models. It has eight state variables, which makes it complicated enough to be interesting without obscuring the main points of the exercise. The eight state variables are: the transmembrane potential ($V$), sodium-channel activation and inactivation gates ($m$ and $h$, similar to the Hodgkin-Huxley model), with an additional slow inactivation gate ($j$), calcium-channel activation and deactivations gates ($d$ and $f$), a time-dependent inward-rectifying potassium current gate ($x_1$), and intracellular calcium concentration ($c$). There are four currents: a sodium current ($i_{Na}$), a calcium current ($i_{Ca}$), and two potassium currents, one time-dependent ($i_{x_1}$) and one background time-independent ($i_{K_1}$).

## CPU-Only Beeler-Reuter Solver

Let's start by developing a CPU only IMEX solver. The main idea is to use the *DifferentialEquations* framework to handle the implicit part of the equation and code the analytical approximation for explicit part separately. If no analytical approximation was known for the explicit part, one could use methods from [this list](https://docs.sciml.ai/latest/solvers/split_ode_solve/#Implicit-Explicit-(IMEX)-ODE-1).

First, we define the model constants:

```julia
const v0 = -84.624
const v1 = 10.0
const C_K1 = 1.0f0
const C_x1 = 1.0f0
const C_Na = 1.0f0
const C_s = 1.0f0
const D_Ca = 0.0f0
const D_Na = 0.0f0
const g_s = 0.09f0
const g_Na = 4.0f0
const g_NaC = 0.005f0
const ENa = 50.0f0 + D_Na
const γ = 0.5f0
const C_m = 1.0f0
```

```
1.0f0
```





Note that the constants are defined as `Float32` and not `Float64`. The reason is that most GPUs have many more single precision cores than double precision ones. To ensure uniformity between CPU and GPU, we also code most states variables as `Float32` except for the transmembrane potential, which is solved by an implicit solver provided by the Sundial library and needs to be `Float64`.

### The State Structure

Next, we define a struct to contain our state. `BeelerReuterCpu` is a functor and we will define a deriv function as its associated function.

```julia
mutable struct BeelerReuterCpu <: Function
    t::Float64              # the last timestep time to calculate Δt
    diff_coef::Float64      # the diffusion-coefficient (coupling strength)

    C::Array{Float32, 2}    # intracellular calcium concentration
    M::Array{Float32, 2}    # sodium current activation gate (m)
    H::Array{Float32, 2}    # sodium current inactivation gate (h)
    J::Array{Float32, 2}    # sodium current slow inactivaiton gate (j)
    D::Array{Float32, 2}    # calcium current activaiton gate (d)
    F::Array{Float32, 2}    # calcium current inactivation gate (f)
    XI::Array{Float32, 2}   # inward-rectifying potassium current (iK1)

    Δu::Array{Float64, 2}   # place-holder for the Laplacian

    function BeelerReuterCpu(u0, diff_coef)
        self = new()

        ny, nx = size(u0)
        self.t = 0.0
        self.diff_coef = diff_coef

        self.C = fill(0.0001f0, (ny,nx))
        self.M = fill(0.01f0, (ny,nx))
        self.H = fill(0.988f0, (ny,nx))
        self.J = fill(0.975f0, (ny,nx))
        self.D = fill(0.003f0, (ny,nx))
        self.F = fill(0.994f0, (ny,nx))
        self.XI = fill(0.0001f0, (ny,nx))

        self.Δu = zeros(ny,nx)

        return self
    end
end
```




### Laplacian

The finite-difference Laplacian is calculated in-place by a 5-point stencil. The Neumann boundary condition is enforced. Note that we could have also used [DiffEqOperators.jl](https://github.com/JuliaDiffEq/DiffEqOperators.jl) to automate this step.

```julia
# 5-point stencil
function laplacian(Δu, u)
    n1, n2 = size(u)

    # internal nodes
    for j = 2:n2-1
        for i = 2:n1-1
            @inbounds  Δu[i,j] = u[i+1,j] + u[i-1,j] + u[i,j+1] + u[i,j-1] - 4*u[i,j]
        end
    end

    # left/right edges
    for i = 2:n1-1
        @inbounds Δu[i,1] = u[i+1,1] + u[i-1,1] + 2*u[i,2] - 4*u[i,1]
        @inbounds Δu[i,n2] = u[i+1,n2] + u[i-1,n2] + 2*u[i,n2-1] - 4*u[i,n2]
    end

    # top/bottom edges
    for j = 2:n2-1
        @inbounds Δu[1,j] = u[1,j+1] + u[1,j-1] + 2*u[2,j] - 4*u[1,j]
        @inbounds Δu[n1,j] = u[n1,j+1] + u[n1,j-1] + 2*u[n1-1,j] - 4*u[n1,j]
    end

    # corners
    @inbounds Δu[1,1] = 2*(u[2,1] + u[1,2]) - 4*u[1,1]
    @inbounds Δu[n1,1] = 2*(u[n1-1,1] + u[n1,2]) - 4*u[n1,1]
    @inbounds Δu[1,n2] = 2*(u[2,n2] + u[1,n2-1]) - 4*u[1,n2]
    @inbounds Δu[n1,n2] = 2*(u[n1-1,n2] + u[n1,n2-1]) - 4*u[n1,n2]
end
```

```
laplacian (generic function with 1 method)
```





### The Rush-Larsen Method

We use an explicit solver for all the state variables except for the transmembrane potential which is solved with the help of an implicit solver. The explicit solver is a domain-specific exponential method, the Rush-Larsen method. This method utilizes an approximation on the model in order to transform the IMEX equation into a form suitable for an implicit ODE solver. This combination of implicit and explicit methods forms a specialized IMEX solver. For general IMEX integration, please see the [IMEX solvers documentation](https://docs.sciml.ai/latest/solvers/split_ode_solve/#Implicit-Explicit-(IMEX)-ODE-1). While we could have used the general model to solve the current problem, for this specific model, the transformation approach is more efficient and is of practical interest.

The [Rush-Larsen](https://ieeexplore.ieee.org/document/4122859/) method replaces the explicit Euler integration for the gating variables with direct integration. The starting point is the general ODE for the gating variables in Hodgkin-Huxley style ODEs,

$$\frac{dg}{dt} = \alpha(V) (1 - g) - \beta(V) g$$

where $g$ is a generic gating variable, ranging from 0 to 1, and $\alpha$ and $\beta$ are reaction rates. This equation can be written as,

$$\frac{dg}{dt} = (g_{\infty} - g) / \tau_g,$$

where $g_\infty$ and $\tau_g$ are

$$g_{\infty} = \frac{\alpha}{(\alpha + \beta)},$$

and,

$$\tau_g = \frac{1}{(\alpha + \beta)}.$$

Assuing that $g_\infty$ and $\tau_g$ are constant for the duration of a single time step ($\Delta{t}$), which is a reasonable assumption for most cardiac models, we can integrate directly to have,

$$g(t + \Delta{t}) = g_{\infty} - \left(g_{\infty} - g(\Delta{t})\right)\,e^{-\Delta{t}/\tau_g}.$$

This is the Rush-Larsen technique. Note that as $\Delta{t} \rightarrow 0$, this equations morphs into the explicit Euler formula,

$$g(t + \Delta{t}) = g(t) + \Delta{t}\frac{dg}{dt}.$$

`rush_larsen` is a helper function that use the Rush-Larsen method to integrate the gating variables.

```julia
@inline function rush_larsen(g, α, β, Δt)
    inf = α/(α+β)
    τ = 1f0 / (α+β)
    return clamp(g + (g - inf) * expm1(-Δt/τ), 0f0, 1f0)
end
```

```
rush_larsen (generic function with 1 method)
```





The gating variables are updated as below. The details of how to calculate $\alpha$ and $\beta$ are based on the Beeler-Reuter model and not of direct interest to this tutorial.

```julia
function update_M_cpu(g, v, Δt)
    # the condition is needed here to prevent NaN when v == 47.0
    α = isapprox(v, 47.0f0) ? 10.0f0 : -(v+47.0f0) / (exp(-0.1f0*(v+47.0f0)) - 1.0f0)
    β = (40.0f0 * exp(-0.056f0*(v+72.0f0)))
    return rush_larsen(g, α, β, Δt)
end

function update_H_cpu(g, v, Δt)
    α = 0.126f0 * exp(-0.25f0*(v+77.0f0))
    β = 1.7f0 / (exp(-0.082f0*(v+22.5f0)) + 1.0f0)
   return rush_larsen(g, α, β, Δt)
end

function update_J_cpu(g, v, Δt)
    α = (0.55f0 * exp(-0.25f0*(v+78.0f0))) / (exp(-0.2f0*(v+78.0f0)) + 1.0f0)
    β = 0.3f0 / (exp(-0.1f0*(v+32.0f0)) + 1.0f0)
    return rush_larsen(g, α, β, Δt)
end

function update_D_cpu(g, v, Δt)
    α = γ * (0.095f0 * exp(-0.01f0*(v-5.0f0))) / (exp(-0.072f0*(v-5.0f0)) + 1.0f0)
    β = γ * (0.07f0 * exp(-0.017f0*(v+44.0f0))) / (exp(0.05f0*(v+44.0f0)) + 1.0f0)
    return rush_larsen(g, α, β, Δt)
end

function update_F_cpu(g, v, Δt)
    α = γ * (0.012f0 * exp(-0.008f0*(v+28.0f0))) / (exp(0.15f0*(v+28.0f0)) + 1.0f0)
    β = γ * (0.0065f0 * exp(-0.02f0*(v+30.0f0))) / (exp(-0.2f0*(v+30.0f0)) + 1.0f0)
    return rush_larsen(g, α, β, Δt)
end

function update_XI_cpu(g, v, Δt)
    α = (0.0005f0 * exp(0.083f0*(v+50.0f0))) / (exp(0.057f0*(v+50.0f0)) + 1.0f0)
    β = (0.0013f0 * exp(-0.06f0*(v+20.0f0))) / (exp(-0.04f0*(v+20.0f0)) + 1.0f0)
    return rush_larsen(g, α, β, Δt)
end
```

```
update_XI_cpu (generic function with 1 method)
```





The intracelleular calcium is not technically a gating variable, but we can use a similar explicit exponential integrator for it.

```julia
function update_C_cpu(g, d, f, v, Δt)
    ECa = D_Ca - 82.3f0 - 13.0278f0 * log(g)
    kCa = C_s * g_s * d * f
    iCa = kCa * (v - ECa)
    inf = 1.0f-7 * (0.07f0 - g)
    τ = 1f0 / 0.07f0
    return g + (g - inf) * expm1(-Δt/τ)
end
```

```
update_C_cpu (generic function with 1 method)
```





### Implicit Solver

Now, it is time to define the derivative function as an associated function of **BeelerReuterCpu**. We plan to use the CVODE_BDF solver as our implicit portion. Similar to other iterative methods, it calls the deriv function with the same $t$ multiple times. For example, these are consecutive $t$s from a representative run:

0.86830
0.86830
0.85485
0.85485
0.85485
0.86359
0.86359
0.86359
0.87233
0.87233
0.87233
0.88598
...

Here, every time step is called three times. We distinguish between two types of calls to the deriv function. When $t$ changes, the gating variables are updated by calling `update_gates_cpu`:

```julia
function update_gates_cpu(u, XI, M, H, J, D, F, C, Δt)
    let Δt = Float32(Δt)
        n1, n2 = size(u)
        for j = 1:n2
            for i = 1:n1
                v = Float32(u[i,j])

                XI[i,j] = update_XI_cpu(XI[i,j], v, Δt)
                M[i,j] = update_M_cpu(M[i,j], v, Δt)
                H[i,j] = update_H_cpu(H[i,j], v, Δt)
                J[i,j] = update_J_cpu(J[i,j], v, Δt)
                D[i,j] = update_D_cpu(D[i,j], v, Δt)
                F[i,j] = update_F_cpu(F[i,j], v, Δt)

                C[i,j] = update_C_cpu(C[i,j], D[i,j], F[i,j], v, Δt)
            end
        end
    end
end
```

```
update_gates_cpu (generic function with 1 method)
```





On the other hand, du is updated at each time step, since it is independent of $\Delta{t}$.

```julia
# iK1 is the inward-rectifying potassium current
function calc_iK1(v)
    ea = exp(0.04f0*(v+85f0))
    eb = exp(0.08f0*(v+53f0))
    ec = exp(0.04f0*(v+53f0))
    ed = exp(-0.04f0*(v+23f0))
    return 0.35f0 * (4f0*(ea-1f0)/(eb + ec)
            + 0.2f0 * (isapprox(v, -23f0) ? 25f0 : (v+23f0) / (1f0-ed)))
end

# ix1 is the time-independent background potassium current
function calc_ix1(v, xi)
    ea = exp(0.04f0*(v+77f0))
    eb = exp(0.04f0*(v+35f0))
    return xi * 0.8f0 * (ea-1f0) / eb
end

# iNa is the sodium current (similar to the classic Hodgkin-Huxley model)
function calc_iNa(v, m, h, j)
    return C_Na * (g_Na * m^3 * h * j + g_NaC) * (v - ENa)
end

# iCa is the calcium current
function calc_iCa(v, d, f, c)
    ECa = D_Ca - 82.3f0 - 13.0278f0 * log(c)    # ECa is the calcium reversal potential
    return C_s * g_s * d * f * (v - ECa)
end

function update_du_cpu(du, u, XI, M, H, J, D, F, C)
    n1, n2 = size(u)

    for j = 1:n2
        for i = 1:n1
            v = Float32(u[i,j])

            # calculating individual currents
            iK1 = calc_iK1(v)
            ix1 = calc_ix1(v, XI[i,j])
            iNa = calc_iNa(v, M[i,j], H[i,j], J[i,j])
            iCa = calc_iCa(v, D[i,j], F[i,j], C[i,j])

            # total current
            I_sum = iK1 + ix1 + iNa + iCa

            # the reaction part of the reaction-diffusion equation
            du[i,j] = -I_sum / C_m
        end
    end
end
```

```
update_du_cpu (generic function with 1 method)
```





Finally, we put everything together is our deriv function, which is a call on `BeelerReuterCpu`.

```julia
function (f::BeelerReuterCpu)(du, u, p, t)
    Δt = t - f.t

    if Δt != 0 || t == 0
        update_gates_cpu(u, f.XI, f.M, f.H, f.J, f.D, f.F, f.C, Δt)
        f.t = t
    end

    laplacian(f.Δu, u)

    # calculate the reaction portion
    update_du_cpu(du, u, f.XI, f.M, f.H, f.J, f.D, f.F, f.C)

    # ...add the diffusion portion
    du .+= f.diff_coef .* f.Δu
end
```




### Results

Time to test! We need to define the starting transmembrane potential with the help of global constants **v0** and **v1**, which represent the resting and activated potentials.

```julia
const N = 192;
u0 = fill(v0, (N, N));
u0[90:102,90:102] .= v1;   # a small square in the middle of the domain
```




The initial condition is a small square in the middle of the domain.

```julia
using Plots
heatmap(u0)
```

![](figures/01-beeler_reuter_11_1.png)



Next, the problem is defined:

```julia
using DifferentialEquations, Sundials

deriv_cpu = BeelerReuterCpu(u0, 1.0);
prob = ODEProblem(deriv_cpu, u0, (0.0, 50.0));
```




For stiff reaction-diffusion equations, CVODE_BDF from Sundial library is an excellent solver.

```julia
@time sol = solve(prob, CVODE_BDF(linear_solver=:GMRES), saveat=100.0);
```

```
30.456944 seconds (1.97 M allocations: 122.018 MiB, 0.10% gc time, 3.29% c
ompilation time)
```



```julia
heatmap(sol.u[end])
```

![](figures/01-beeler_reuter_14_1.png)



## CPU/GPU Beeler-Reuter Solver

GPUs are great for embarrassingly parallel problems but not so much for highly coupled models. We plan to keep the implicit part on CPU and run the decoupled explicit code on a GPU with the help of the CUDAnative library.

### GPUs and CUDA

It this section, we present a brief summary of how GPUs (specifically NVIDIA GPUs) work and how to program them using the Julia CUDA interface. The readers who are familiar with these basic concepts may skip this section.

Let's start by looking at the hardware of a typical high-end GPU, GTX 1080. It has four Graphics Processing Clusters (equivalent to a discrete CPU), each harboring five Streaming Multiprocessor (similar to a CPU core). Each SM has 128 single-precision CUDA cores. Therefore, GTX 1080 has a total of 4 x 5 x 128 = 2560 CUDA cores. The maximum  theoretical throughput for a GTX 1080 is reported as 8.87 TFLOPS. This figure is calculated for a boost clock frequency of 1.733 MHz as 2 x 2560 x 1.733 MHz = 8.87 TFLOPS. The factor 2 is included because two single floating point operations, a multiplication and an addition, can be done in a clock cycle as part of a fused-multiply-addition FMA operation. GTX 1080 also has 8192 MB of global memory accessible to all the cores (in addition to local and shared memory on each SM).

A typical CUDA application has the following flow:

1. Define and initialize the problem domain tensors (multi-dimensional arrays) in CPU memory.
2. Allocate corresponding tensors in the GPU global memory.
3. Transfer the input tensors from CPU to the corresponding GPU tensors.
4. Invoke CUDA kernels (i.e., the GPU functions callable from CPU) that operate on the GPU tensors.
5. Transfer the result tensors from GPU back to CPU.
6. Process tensors on CPU.
7. Repeat steps 3-6 as needed.

Some libraries, such as [ArrayFire](https://github.com/arrayfire/arrayfire), hide the complexicities of steps 2-5 behind a higher level of abstraction. However, here we take a lower level route. By using [CUDA](https://github.com/JuliaGPU/CUDA.jl), we achieve a finer-grained control and higher performance. In return, we need to implement each step manually.

*CuArray* is a thin abstraction layer over the CUDA API and allows us to define GPU-side tensors and copy data to and from them but does not provide for operations on tensors. *CUDAnative* is a compiler that translates Julia functions designated as CUDA kernels into ptx (a high-level CUDA assembly language).

### The CUDA Code

The key to fast CUDA programs is to minimize CPU/GPU memory transfers and global memory accesses. The implicit solver is currently CPU only, but it only needs access to the transmembrane potential. The rest of state variables reside on the GPU memory.

We modify ``BeelerReuterCpu`` into ``BeelerReuterGpu`` by defining the state variables as *CuArray*s instead of standard Julia *Array*s. The name of each variable defined on GPU is prefixed by *d_* for clarity. Note that $\Delta{v}$ is a temporary storage for the Laplacian and stays on the CPU side.

```julia
using CUDA

mutable struct BeelerReuterGpu <: Function
    t::Float64                  # the last timestep time to calculate Δt
    diff_coef::Float64          # the diffusion-coefficient (coupling strength)

    d_C::CuArray{Float32, 2}    # intracellular calcium concentration
    d_M::CuArray{Float32, 2}    # sodium current activation gate (m)
    d_H::CuArray{Float32, 2}    # sodium current inactivation gate (h)
    d_J::CuArray{Float32, 2}    # sodium current slow inactivaiton gate (j)
    d_D::CuArray{Float32, 2}    # calcium current activaiton gate (d)
    d_F::CuArray{Float32, 2}    # calcium current inactivation gate (f)
    d_XI::CuArray{Float32, 2}   # inward-rectifying potassium current (iK1)

    d_u::CuArray{Float64, 2}    # place-holder for u in the device memory
    d_du::CuArray{Float64, 2}   # place-holder for d_u in the device memory

    Δv::Array{Float64, 2}       # place-holder for voltage gradient

    function BeelerReuterGpu(u0, diff_coef)
        self = new()

        ny, nx = size(u0)
        @assert (nx % 16 == 0) && (ny % 16 == 0)
        self.t = 0.0
        self.diff_coef = diff_coef

        self.d_C = CuArray(fill(0.0001f0, (ny,nx)))
        self.d_M = CuArray(fill(0.01f0, (ny,nx)))
        self.d_H = CuArray(fill(0.988f0, (ny,nx)))
        self.d_J = CuArray(fill(0.975f0, (ny,nx)))
        self.d_D = CuArray(fill(0.003f0, (ny,nx)))
        self.d_F = CuArray(fill(0.994f0, (ny,nx)))
        self.d_XI = CuArray(fill(0.0001f0, (ny,nx)))

        self.d_u = CuArray(u0)
        self.d_du = CuArray(zeros(ny,nx))

        self.Δv = zeros(ny,nx)

        return self
    end
end
```




The Laplacian function remains unchanged. The main change to the explicit gating solvers is that *exp* and *expm1* functions are prefixed by *CUDAnative.*. This is a technical nuisance that will hopefully be resolved in future.

```julia
function rush_larsen_gpu(g, α, β, Δt)
    inf = α/(α+β)
    τ = 1.0/(α+β)
    return clamp(g + (g - inf) * CUDAnative.expm1(-Δt/τ), 0f0, 1f0)
end

function update_M_gpu(g, v, Δt)
    # the condition is needed here to prevent NaN when v == 47.0
    α = isapprox(v, 47.0f0) ? 10.0f0 : -(v+47.0f0) / (CUDAnative.exp(-0.1f0*(v+47.0f0)) - 1.0f0)
    β = (40.0f0 * CUDAnative.exp(-0.056f0*(v+72.0f0)))
    return rush_larsen_gpu(g, α, β, Δt)
end

function update_H_gpu(g, v, Δt)
    α = 0.126f0 * CUDAnative.exp(-0.25f0*(v+77.0f0))
    β = 1.7f0 / (CUDAnative.exp(-0.082f0*(v+22.5f0)) + 1.0f0)
    return rush_larsen_gpu(g, α, β, Δt)
end

function update_J_gpu(g, v, Δt)
    α = (0.55f0 * CUDAnative.exp(-0.25f0*(v+78.0f0))) / (CUDAnative.exp(-0.2f0*(v+78.0f0)) + 1.0f0)
    β = 0.3f0 / (CUDAnative.exp(-0.1f0*(v+32.0f0)) + 1.0f0)
    return rush_larsen_gpu(g, α, β, Δt)
end

function update_D_gpu(g, v, Δt)
    α = γ * (0.095f0 * CUDAnative.exp(-0.01f0*(v-5.0f0))) / (CUDAnative.exp(-0.072f0*(v-5.0f0)) + 1.0f0)
    β = γ * (0.07f0 * CUDAnative.exp(-0.017f0*(v+44.0f0))) / (CUDAnative.exp(0.05f0*(v+44.0f0)) + 1.0f0)
    return rush_larsen_gpu(g, α, β, Δt)
end

function update_F_gpu(g, v, Δt)
    α = γ * (0.012f0 * CUDAnative.exp(-0.008f0*(v+28.0f0))) / (CUDAnative.exp(0.15f0*(v+28.0f0)) + 1.0f0)
    β = γ * (0.0065f0 * CUDAnative.exp(-0.02f0*(v+30.0f0))) / (CUDAnative.exp(-0.2f0*(v+30.0f0)) + 1.0f0)
    return rush_larsen_gpu(g, α, β, Δt)
end

function update_XI_gpu(g, v, Δt)
    α = (0.0005f0 * CUDAnative.exp(0.083f0*(v+50.0f0))) / (CUDAnative.exp(0.057f0*(v+50.0f0)) + 1.0f0)
    β = (0.0013f0 * CUDAnative.exp(-0.06f0*(v+20.0f0))) / (CUDAnative.exp(-0.04f0*(v+20.0f0)) + 1.0f0)
    return rush_larsen_gpu(g, α, β, Δt)
end

function update_C_gpu(c, d, f, v, Δt)
    ECa = D_Ca - 82.3f0 - 13.0278f0 * CUDAnative.log(c)
    kCa = C_s * g_s * d * f
    iCa = kCa * (v - ECa)
    inf = 1.0f-7 * (0.07f0 - c)
    τ = 1f0 / 0.07f0
    return c + (c - inf) * CUDAnative.expm1(-Δt/τ)
end
```

```
update_C_gpu (generic function with 1 method)
```





Similarly, we modify the functions to calculate the individual currents by adding CUDAnative prefix.

```julia
# iK1 is the inward-rectifying potassium current
function calc_iK1(v)
    ea = CUDAnative.exp(0.04f0*(v+85f0))
    eb = CUDAnative.exp(0.08f0*(v+53f0))
    ec = CUDAnative.exp(0.04f0*(v+53f0))
    ed = CUDAnative.exp(-0.04f0*(v+23f0))
    return 0.35f0 * (4f0*(ea-1f0)/(eb + ec)
            + 0.2f0 * (isapprox(v, -23f0) ? 25f0 : (v+23f0) / (1f0-ed)))
end

# ix1 is the time-independent background potassium current
function calc_ix1(v, xi)
    ea = CUDAnative.exp(0.04f0*(v+77f0))
    eb = CUDAnative.exp(0.04f0*(v+35f0))
    return xi * 0.8f0 * (ea-1f0) / eb
end

# iNa is the sodium current (similar to the classic Hodgkin-Huxley model)
function calc_iNa(v, m, h, j)
    return C_Na * (g_Na * m^3 * h * j + g_NaC) * (v - ENa)
end

# iCa is the calcium current
function calc_iCa(v, d, f, c)
    ECa = D_Ca - 82.3f0 - 13.0278f0 * CUDAnative.log(c)    # ECa is the calcium reversal potential
    return C_s * g_s * d * f * (v - ECa)
end
```

```
calc_iCa (generic function with 1 method)
```





### CUDA Kernels

A CUDA program does not directly deal with GPCs and SMs. The logical view of a CUDA program is in the term of *blocks* and *threads*. We have to specify the number of block and threads when running a CUDA *kernel*. Each thread runs on a single CUDA core. Threads are logically bundled into blocks, which are in turn specified on a grid. The grid stands for the entirety of the domain of interest.

Each thread can find its logical coordinate by using few pre-defined indexing variables (*threadIdx*, *blockIdx*, *blockDim* and *gridDim*) in C/C++ and the corresponding functions (e.g., `threadIdx()`) in Julia. There variables and functions are defined automatically for each thread and may return a different value depending on the calling thread. The return value of these functions is a 1, 2, or 3 dimensional structure whose elements can be accessed as `.x`, `.y`, and `.z` (for a 1-dimensional case, `.x` reports the actual index and `.y` and `.z` simply return 1). For example, if we deploy a kernel in 128 blocks and with 256 threads per block, each thread will see

```
    gridDim.x = 128;
    blockDim=256;
```

while `blockIdx.x` ranges from 0 to 127 in C/C++ and 1 to 128 in Julia. Similarly, `threadIdx.x` will be between 0 to 255 in C/C++ (of course, in Julia the range will be 1 to 256).

A C/C++ thread can calculate its index as

```
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
```

In Julia, we have to take into account base 1. Therefore, we use the following formula

```
    idx = (blockIdx().x-UInt32(1)) * blockDim().x + threadIdx().x
```

A CUDA programmer is free to interpret the calculated index however it fits the application, but in practice, it is usually interpreted as an index into input tensors.

In the GPU version of the solver, each thread works on a single element of the medium, indexed by a (x,y) pair.
`update_gates_gpu` and `update_du_gpu` are very similar to their CPU counterparts but are in fact CUDA kernels where the *for* loops are replaced with CUDA specific indexing. Note that CUDA kernels cannot return a valve; hence, *nothing* at the end.

```julia
function update_gates_gpu(u, XI, M, H, J, D, F, C, Δt)
    i = (blockIdx().x-UInt32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y-UInt32(1)) * blockDim().y + threadIdx().y

    v = Float32(u[i,j])

    let Δt = Float32(Δt)
        XI[i,j] = update_XI_gpu(XI[i,j], v, Δt)
        M[i,j] = update_M_gpu(M[i,j], v, Δt)
        H[i,j] = update_H_gpu(H[i,j], v, Δt)
        J[i,j] = update_J_gpu(J[i,j], v, Δt)
        D[i,j] = update_D_gpu(D[i,j], v, Δt)
        F[i,j] = update_F_gpu(F[i,j], v, Δt)

        C[i,j] = update_C_gpu(C[i,j], D[i,j], F[i,j], v, Δt)
    end
    nothing
end

function update_du_gpu(du, u, XI, M, H, J, D, F, C)
    i = (blockIdx().x-UInt32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y-UInt32(1)) * blockDim().y + threadIdx().y

    v = Float32(u[i,j])

    # calculating individual currents
    iK1 = calc_iK1(v)
    ix1 = calc_ix1(v, XI[i,j])
    iNa = calc_iNa(v, M[i,j], H[i,j], J[i,j])
    iCa = calc_iCa(v, D[i,j], F[i,j], C[i,j])

    # total current
    I_sum = iK1 + ix1 + iNa + iCa

    # the reaction part of the reaction-diffusion equation
    du[i,j] = -I_sum / C_m
    nothing
end
```

```
update_du_gpu (generic function with 1 method)
```





### Implicit Solver

Finally, the deriv function is modified to copy *u* to GPU and copy *du* back and to invoke CUDA kernels.

```julia
function (f::BeelerReuterGpu)(du, u, p, t)
    L = 16   # block size
    Δt = t - f.t
    copyto!(f.d_u, u)
    ny, nx = size(u)

    if Δt != 0 || t == 0
        @cuda blocks=(ny÷L,nx÷L) threads=(L,L) update_gates_gpu(
            f.d_u, f.d_XI, f.d_M, f.d_H, f.d_J, f.d_D, f.d_F, f.d_C, Δt)
        f.t = t
    end

    laplacian(f.Δv, u)

    # calculate the reaction portion
    @cuda blocks=(ny÷L,nx÷L) threads=(L,L) update_du_gpu(
        f.d_du, f.d_u, f.d_XI, f.d_M, f.d_H, f.d_J, f.d_D, f.d_F, f.d_C)

    copyto!(du, f.d_du)

    # ...add the diffusion portion
    du .+= f.diff_coef .* f.Δv
end
```




Ready to test!

```julia
using DifferentialEquations, Sundials

deriv_gpu = BeelerReuterGpu(u0, 1.0);
prob = ODEProblem(deriv_gpu, u0, (0.0, 50.0));
@time sol = solve(prob, CVODE_BDF(linear_solver=:GMRES), saveat=100.0);
```

```
Error: CUDA.jl did not successfully initialize, and is not usable.
If you did not see any other error message, try again in a new session
with the JULIA_DEBUG environment variable set to 'CUDA'.
```



```julia
heatmap(sol.u[end])
```

![](figures/01-beeler_reuter_21_1.png)



## Summary

We achieve around a 6x speedup with running the explicit portion of our IMEX solver on a GPU. The major bottleneck of this technique is the communication between CPU and GPU. In its current form, not all of the internals of the method utilize GPU acceleration. In particular, the implicit equations solved by GMRES are performed on the CPU. This partial CPU nature also increases the amount of data transfer that is required between the GPU and CPU (performed every f call). Compiling the full ODE solver to the GPU would solve both of these issues and potentially give a much larger speedup. [JuliaDiffEq developers are currently working on solutions to alleviate these issues](http://www.stochasticlifestyle.com/solving-systems-stochastic-pdes-using-gpus-julia/), but these will only be compatible with native Julia solvers (and not Sundials).


## Appendix

These tutorials are a part of the SciMLTutorials.jl repository, found at: [https://github.com/SciML/SciMLTutorials.jl](https://github.com/SciML/SciMLTutorials.jl). For more information on high-performance scientific machine learning, check out the SciML Open Source Software Organization [https://sciml.ai](https://sciml.ai).

To locally run this tutorial, do the following commands:

```
using SciMLTutorials
SciMLTutorials.weave_file("tutorials/advanced","01-beeler_reuter.jmd")
```

Computer Information:

```
Julia Version 1.6.2
Commit 1b93d53fc4 (2021-07-14 15:36 UTC)
Platform Info:
  OS: Linux (x86_64-pc-linux-gnu)
  CPU: AMD EPYC 7502 32-Core Processor
  WORD_SIZE: 64
  LIBM: libopenlibm
  LLVM: libLLVM-11.0.1 (ORCJIT, znver2)
Environment:
  JULIA_DEPOT_PATH = /root/.cache/julia-buildkite-plugin/depots/a6029d3a-f78b-41ea-bc97-28aa57c6c6ea
  JULIA_NUM_THREADS = 16

```

Package Information:

```
      Status `/var/lib/buildkite-agent/builds/6-amdci4-julia-csail-mit-edu/julialang/scimltutorials-dot-jl/tutorials/advanced/Project.toml`
  [2169fc97] AlgebraicMultigrid v0.4.0
  [6e4b80f9] BenchmarkTools v1.0.0
  [052768ef] CUDA v2.6.3
  [2b5f629d] DiffEqBase v6.62.2
  [9fdde737] DiffEqOperators v4.26.0
  [0c46a032] DifferentialEquations v6.17.1
  [587475ba] Flux v0.12.1
  [961ee093] ModelingToolkit v5.17.3
  [2774e3e8] NLsolve v4.5.1
  [315f7962] NeuralPDE v3.10.1
  [1dea7af3] OrdinaryDiffEq v5.56.0
  [91a5bcdd] Plots v1.15.2
  [0bca4576] SciMLBase v1.13.4
  [30cb0354] SciMLTutorials v0.9.0
  [47a9eef4] SparseDiffTools v1.13.2
  [684fba80] SparsityDetection v0.3.4
  [789caeaf] StochasticDiffEq v6.34.1
  [c3572dad] Sundials v4.4.3
  [37e2e46d] LinearAlgebra
  [2f01184e] SparseArrays
```

And the full manifest:

```
      Status `/var/lib/buildkite-agent/builds/6-amdci4-julia-csail-mit-edu/julialang/scimltutorials-dot-jl/tutorials/advanced/Manifest.toml`
  [c3fe647b] AbstractAlgebra v0.16.0
  [621f4979] AbstractFFTs v1.0.1
  [1520ce14] AbstractTrees v0.3.4
  [79e6a3ab] Adapt v3.3.0
  [2169fc97] AlgebraicMultigrid v0.4.0
  [ec485272] ArnoldiMethod v0.1.0
  [4fba245c] ArrayInterface v3.1.15
  [4c555306] ArrayLayouts v0.7.0
  [13072b0f] AxisAlgorithms v1.0.0
  [ab4f0b2a] BFloat16s v0.1.0
  [aae01518] BandedMatrices v0.16.9
  [6e4b80f9] BenchmarkTools v1.0.0
  [8e7c35d0] BlockArrays v0.15.3
  [ffab5731] BlockBandedMatrices v0.10.6
  [764a87c0] BoundaryValueDiffEq v2.7.1
  [fa961155] CEnum v0.4.1
  [00ebfdb7] CSTParser v2.5.0
  [052768ef] CUDA v2.6.3
  [7057c7e9] Cassette v0.3.6
  [082447d4] ChainRules v0.7.65
  [d360d2e6] ChainRulesCore v0.9.44
  [b630d9fa] CheapThreads v0.2.5
  [944b1d66] CodecZlib v0.7.0
  [35d6a980] ColorSchemes v3.12.1
  [3da002f7] ColorTypes v0.11.0
  [5ae59095] Colors v0.12.8
  [861a8166] Combinatorics v1.0.2
  [a80b9123] CommonMark v0.8.1
  [38540f10] CommonSolve v0.2.0
  [bbf7d656] CommonSubexpressions v0.3.0
  [34da2185] Compat v3.30.0
  [aa819f21] CompatHelper v1.18.6
  [8f4d0f93] Conda v1.5.2
  [88cd18e8] ConsoleProgressMonitor v0.1.2
  [187b0558] ConstructionBase v1.2.1
  [d38c429a] Contour v0.5.7
  [a8cc5b0e] Crayons v4.0.4
  [8a292aeb] Cuba v2.2.0
  [667455a9] Cubature v1.5.1
  [9a962f9c] DataAPI v1.6.0
  [82cc6244] DataInterpolations v3.3.1
  [864edb3b] DataStructures v0.18.9
  [e2d170a0] DataValueInterfaces v1.0.0
  [bcd4f6db] DelayDiffEq v5.31.0
  [2b5f629d] DiffEqBase v6.62.2
  [459566f4] DiffEqCallbacks v2.16.1
  [5a0ffddc] DiffEqFinancial v2.4.0
  [aae7a2af] DiffEqFlux v1.37.0
  [c894b116] DiffEqJump v6.14.2
  [77a26b50] DiffEqNoiseProcess v5.7.3
  [9fdde737] DiffEqOperators v4.26.0
  [055956cb] DiffEqPhysics v3.9.0
  [41bf760c] DiffEqSensitivity v6.45.0
  [163ba53b] DiffResults v1.0.3
  [b552c78f] DiffRules v1.0.2
  [0c46a032] DifferentialEquations v6.17.1
  [c619ae07] DimensionalPlotRecipes v1.2.0
  [b4f34e82] Distances v0.10.3
  [31c24e10] Distributions v0.24.18
  [ced4e74d] DistributionsAD v0.6.26
  [ffbed154] DocStringExtensions v0.8.4
  [e30172f5] Documenter v0.26.3
  [d4d017d3] ExponentialUtilities v1.8.4
  [e2ba6199] ExprTools v0.1.3
  [8f5d6c58] EzXML v1.1.0
  [c87230d0] FFMPEG v0.4.0
  [7a1cc6ca] FFTW v1.4.1
  [7034ab61] FastBroadcast v0.1.8
  [9aa1b823] FastClosures v0.3.2
  [1a297f60] FillArrays v0.11.7
  [6a86dc24] FiniteDiff v2.8.0
  [53c48c17] FixedPointNumbers v0.8.4
  [587475ba] Flux v0.12.1
  [59287772] Formatting v0.4.2
  [f6369f11] ForwardDiff v0.10.18
  [069b7b12] FunctionWrappers v1.1.2
  [d9f16b24] Functors v0.2.1
  [0c68f7d7] GPUArrays v6.4.1
  [61eb1bfa] GPUCompiler v0.10.0
  [28b8d3ca] GR v0.57.4
  [a75be94c] GalacticOptim v1.2.0
  [5c1252a2] GeometryBasics v0.3.12
  [bc5e4493] GitHub v5.4.0
  [af5da776] GlobalSensitivity v1.0.0
  [42e2da0e] Grisu v1.0.2
  [19dc6840] HCubature v1.5.0
  [cd3eb016] HTTP v0.9.9
  [eafb193a] Highlights v0.4.5
  [0e44f5e4] Hwloc v2.0.0
  [7073ff75] IJulia v1.23.2
  [b5f81e59] IOCapture v0.1.1
  [7869d1d1] IRTools v0.4.2
  [615f187c] IfElse v0.1.0
  [d25df0c9] Inflate v0.1.2
  [83e8ac13] IniFile v0.5.0
  [a98d9a8b] Interpolations v0.13.2
  [c8e1da08] IterTools v1.3.0
  [42fd0dbc] IterativeSolvers v0.9.1
  [82899510] IteratorInterfaceExtensions v1.0.0
  [692b3bcd] JLLWrappers v1.3.0
  [682c06a0] JSON v0.21.1
  [98e50ef6] JuliaFormatter v0.13.7
  [e5e0dc1b] Juno v0.8.4
  [5ab0869b] KernelDensity v0.6.3
  [929cbde3] LLVM v3.7.1
  [b964fa9f] LaTeXStrings v1.2.1
  [2ee39098] LabelledArrays v1.6.1
  [23fbe1c1] Latexify v0.15.5
  [a5e1c1ea] LatinHypercubeSampling v1.8.0
  [73f95e8e] LatticeRules v0.0.1
  [5078a376] LazyArrays v0.21.4
  [d7e5e226] LazyBandedMatrices v0.5.7
  [1d6d02ad] LeftChildRightSiblingTrees v0.1.2
  [093fc24a] LightGraphs v1.3.5
  [d3d80556] LineSearches v7.1.1
  [2ab3a3ac] LogExpFunctions v0.2.4
  [e6f89c97] LoggingExtras v0.4.6
  [bdcacae8] LoopVectorization v0.12.23
  [1914dd2f] MacroTools v0.5.6
  [a3b82374] MatrixFactorizations v0.8.3
  [739be429] MbedTLS v1.0.3
  [442fdcdd] Measures v0.3.1
  [e89f7d12] Media v0.5.0
  [c03570c3] Memoize v0.4.4
  [e1d29d7a] Missings v1.0.0
  [78c3b35d] Mocking v0.7.1
  [961ee093] ModelingToolkit v5.17.3
  [4886b29c] MonteCarloIntegration v0.0.2
  [46d2c3a1] MuladdMacro v0.2.2
  [f9640e96] MultiScaleArrays v1.8.1
  [ffc61752] Mustache v1.0.10
  [d41bc354] NLSolversBase v7.8.0
  [2774e3e8] NLsolve v4.5.1
  [872c559c] NNlib v0.7.19
  [77ba4419] NaNMath v0.3.5
  [315f7962] NeuralPDE v3.10.1
  [8913a72c] NonlinearSolve v0.3.8
  [6fe1bfb0] OffsetArrays v1.9.0
  [429524aa] Optim v1.3.0
  [bac558e1] OrderedCollections v1.4.1
  [1dea7af3] OrdinaryDiffEq v5.56.0
  [90014a1f] PDMats v0.11.0
  [65888b18] ParameterizedFunctions v5.10.0
  [d96e819e] Parameters v0.12.2
  [69de0a69] Parsers v1.1.0
  [ccf2f8ad] PlotThemes v2.0.1
  [995b91a9] PlotUtils v1.0.10
  [91a5bcdd] Plots v1.15.2
  [e409e4f3] PoissonRandom v0.4.0
  [f517fe37] Polyester v0.3.1
  [85a6dd25] PositiveFactorizations v0.2.4
  [21216c6a] Preferences v1.2.2
  [33c8b6b6] ProgressLogging v0.1.4
  [92933f4c] ProgressMeter v1.6.2
  [1fd47b50] QuadGK v2.4.1
  [67601950] Quadrature v1.8.1
  [8a4e6c94] QuasiMonteCarlo v0.2.2
  [74087812] Random123 v1.3.1
  [fb686558] RandomExtensions v0.4.3
  [e6cf234a] RandomNumbers v1.4.0
  [c84ed2f1] Ratios v0.4.0
  [3cdcf5f2] RecipesBase v1.1.1
  [01d81517] RecipesPipeline v0.3.2
  [731186ca] RecursiveArrayTools v2.11.4
  [f2c3362d] RecursiveFactorization v0.1.12
  [189a3867] Reexport v1.0.0
  [ae029012] Requires v1.1.3
  [ae5879a3] ResettableStacks v1.1.0
  [37e2e3b7] ReverseDiff v1.9.0
  [79098fc4] Rmath v0.7.0
  [7e49a35a] RuntimeGeneratedFunctions v0.5.2
  [476501e8] SLEEFPirates v0.6.20
  [1bc83da4] SafeTestsets v0.0.1
  [0bca4576] SciMLBase v1.13.4
  [30cb0354] SciMLTutorials v0.9.0
  [6c6a2e73] Scratch v1.0.3
  [efcf1570] Setfield v0.7.0
  [992d4aef] Showoff v1.0.3
  [699a6c99] SimpleTraits v0.9.3
  [ed01d8cd] Sobol v1.5.0
  [2133526b] SodiumSeal v0.1.1
  [b85f4697] SoftGlobalScope v1.1.0
  [a2af1166] SortingAlgorithms v1.0.0
  [47a9eef4] SparseDiffTools v1.13.2
  [684fba80] SparsityDetection v0.3.4
  [276daf66] SpecialFunctions v1.4.1
  [860ef19b] StableRNGs v1.0.0
  [aedffcd0] Static v0.2.4
  [90137ffa] StaticArrays v1.2.0
  [82ae8749] StatsAPI v1.0.0
  [2913bbd2] StatsBase v0.33.8
  [4c63d2b9] StatsFuns v0.9.8
  [9672c7b4] SteadyStateDiffEq v1.6.2
  [789caeaf] StochasticDiffEq v6.34.1
  [7792a7ef] StrideArraysCore v0.1.11
  [09ab397b] StructArrays v0.5.1
  [c3572dad] Sundials v4.4.3
  [d1185830] SymbolicUtils v0.11.2
  [0c5d862f] Symbolics v0.1.25
  [3783bdb8] TableTraits v1.0.1
  [bd369af6] Tables v1.4.2
  [5d786b92] TerminalLoggers v0.1.3
  [8290d209] ThreadingUtilities v0.4.4
  [f269a46b] TimeZones v1.5.5
  [a759f4b9] TimerOutputs v0.5.9
  [0796e94c] Tokenize v0.5.16
  [9f7883ad] Tracker v0.2.16
  [3bb67fe8] TranscodingStreams v0.9.5
  [592b5752] Trapz v2.0.2
  [a2a6695c] TreeViews v0.3.0
  [5c2747f8] URIs v1.3.0
  [3a884ed6] UnPack v1.0.2
  [1986cc42] Unitful v1.7.0
  [3d5dd08c] VectorizationBase v0.20.11
  [81def892] VersionParsing v1.2.0
  [19fa3120] VertexSafeGraphs v0.1.2
  [44d3d7a6] Weave v0.10.8
  [efce3f68] WoodburyMatrices v0.5.3
  [ddb6d928] YAML v0.4.6
  [c2297ded] ZMQ v1.2.1
  [a5390f91] ZipFile v0.9.3
  [e88e6eb3] Zygote v0.6.11
  [700de1a5] ZygoteRules v0.2.1
  [6e34b625] Bzip2_jll v1.0.6+5
  [83423d85] Cairo_jll v1.16.0+6
  [3bed1096] Cuba_jll v4.2.1+0
  [7bc98958] Cubature_jll v1.0.4+0
  [5ae413db] EarCut_jll v2.1.5+1
  [2e619515] Expat_jll v2.2.10+0
  [b22a6f82] FFMPEG_jll v4.3.1+4
  [f5851436] FFTW_jll v3.3.9+7
  [a3f928ae] Fontconfig_jll v2.13.1+14
  [d7e528f0] FreeType2_jll v2.10.1+5
  [559328eb] FriBidi_jll v1.0.5+6
  [0656b61e] GLFW_jll v3.3.4+0
  [d2c73de3] GR_jll v0.57.2+0
  [78b55507] Gettext_jll v0.21.0+0
  [7746bdde] Glib_jll v2.68.1+0
  [e33a78d0] Hwloc_jll v2.4.1+0
  [1d5cc7b8] IntelOpenMP_jll v2018.0.3+2
  [aacddb02] JpegTurbo_jll v2.0.1+3
  [c1c5ebd0] LAME_jll v3.100.0+3
  [dd4b983a] LZO_jll v2.10.1+0
  [dd192d2f] LibVPX_jll v1.9.0+1
  [e9f186c6] Libffi_jll v3.2.2+0
  [d4300ac3] Libgcrypt_jll v1.8.7+0
  [7e76a0d4] Libglvnd_jll v1.3.0+3
  [7add5ba3] Libgpg_error_jll v1.42.0+0
  [94ce4f54] Libiconv_jll v1.16.1+0
  [4b2f31a3] Libmount_jll v2.35.0+0
  [89763e89] Libtiff_jll v4.1.0+2
  [38a345b3] Libuuid_jll v2.36.0+0
  [856f044c] MKL_jll v2021.1.1+1
  [e7412a2a] Ogg_jll v1.3.4+2
  [458c3c95] OpenSSL_jll v1.1.1+6
  [efe28fd5] OpenSpecFun_jll v0.5.4+0
  [91d4177d] Opus_jll v1.3.1+3
  [2f80f16e] PCRE_jll v8.44.0+0
  [30392449] Pixman_jll v0.40.1+0
  [ea2cea3b] Qt5Base_jll v5.15.2+0
  [f50d1b31] Rmath_jll v0.3.0+0
  [fb77eaff] Sundials_jll v5.2.0+1
  [a2964d1f] Wayland_jll v1.17.0+4
  [2381bf8a] Wayland_protocols_jll v1.18.0+4
  [02c8fc9c] XML2_jll v2.9.12+0
  [aed1982a] XSLT_jll v1.1.34+0
  [4f6342f7] Xorg_libX11_jll v1.6.9+4
  [0c0b7dd1] Xorg_libXau_jll v1.0.9+4
  [935fb764] Xorg_libXcursor_jll v1.2.0+4
  [a3789734] Xorg_libXdmcp_jll v1.1.3+4
  [1082639a] Xorg_libXext_jll v1.3.4+4
  [d091e8ba] Xorg_libXfixes_jll v5.0.3+4
  [a51aa0fd] Xorg_libXi_jll v1.7.10+4
  [d1454406] Xorg_libXinerama_jll v1.1.4+4
  [ec84b674] Xorg_libXrandr_jll v1.5.2+4
  [ea2f1a96] Xorg_libXrender_jll v0.9.10+4
  [14d82f49] Xorg_libpthread_stubs_jll v0.1.0+3
  [c7cfdc94] Xorg_libxcb_jll v1.13.0+3
  [cc61e674] Xorg_libxkbfile_jll v1.1.0+4
  [12413925] Xorg_xcb_util_image_jll v0.4.0+1
  [2def613f] Xorg_xcb_util_jll v0.4.0+1
  [975044d2] Xorg_xcb_util_keysyms_jll v0.4.0+1
  [0d47668e] Xorg_xcb_util_renderutil_jll v0.3.9+1
  [c22f9ab0] Xorg_xcb_util_wm_jll v0.4.1+1
  [35661453] Xorg_xkbcomp_jll v1.4.2+4
  [33bec58e] Xorg_xkeyboard_config_jll v2.27.0+4
  [c5fb5394] Xorg_xtrans_jll v1.4.0+3
  [8f1865be] ZeroMQ_jll v4.3.2+6
  [3161d3a3] Zstd_jll v1.5.0+0
  [0ac62f75] libass_jll v0.14.0+4
  [f638f0a6] libfdk_aac_jll v0.1.6+4
  [b53b4c65] libpng_jll v1.6.38+0
  [a9144af2] libsodium_jll v1.0.20+0
  [f27f6e37] libvorbis_jll v1.3.6+6
  [1270edf5] x264_jll v2020.7.14+2
  [dfaa095f] x265_jll v3.0.0+3
  [d8fb68d0] xkbcommon_jll v0.9.1+5
  [0dad84c5] ArgTools
  [56f22d72] Artifacts
  [2a0f44e3] Base64
  [ade2ca70] Dates
  [8bb1440f] DelimitedFiles
  [8ba89e20] Distributed
  [f43a241f] Downloads
  [7b1f6079] FileWatching
  [9fa8497b] Future
  [b77e0a4c] InteractiveUtils
  [4af54fe1] LazyArtifacts
  [b27032c2] LibCURL
  [76f85450] LibGit2
  [8f399da3] Libdl
  [37e2e46d] LinearAlgebra
  [56ddb016] Logging
  [d6f4376e] Markdown
  [a63ad114] Mmap
  [ca575930] NetworkOptions
  [44cfe95a] Pkg
  [de0858da] Printf
  [9abbd945] Profile
  [3fa0cd96] REPL
  [9a3f8284] Random
  [ea8e919c] SHA
  [9e88b42a] Serialization
  [1a1011a3] SharedArrays
  [6462fe0b] Sockets
  [2f01184e] SparseArrays
  [10745b16] Statistics
  [4607b0f0] SuiteSparse
  [fa267f1f] TOML
  [a4e569a6] Tar
  [8dfed614] Test
  [cf7118a7] UUIDs
  [4ec0a83e] Unicode
  [e66e0078] CompilerSupportLibraries_jll
  [deac9b47] LibCURL_jll
  [29816b5a] LibSSH2_jll
  [c8ffd9c3] MbedTLS_jll
  [14a3606d] MozillaCACerts_jll
  [4536629a] OpenBLAS_jll
  [bea87d4a] SuiteSparse_jll
  [83775a58] Zlib_jll
  [8e850ede] nghttp2_jll
  [3f19e933] p7zip_jll
```

