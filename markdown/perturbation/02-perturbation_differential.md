---
author: "Shahriar Iravanian"
title: "Mixed Symbolic/Numerical Methods for Perturbation Theory - Differential Equations"
---


## Prelims

In the previous tutorial, *Mixed Symbolic/Numerical Methods for Perturbation Theory - Algebraic Equations*, we discussed how to solve algebraic equations using **Symbolics.jl**. Here, our goal is to extend the method to differential equations. First, we import the following helper functions that were introduced in *Mixed Symbolic/Numerical Methods for Perturbation Theory - Algebraic Equations*.

```julia
using Symbolics, SymbolicUtils

def_taylor(x, ps) = sum([a*x^i for (i,a) in enumerate(ps)])
def_taylor(x, ps, p₀) = p₀ + def_taylor(x, ps)

function collect_powers(eq, x, ns; max_power=100)
    eq = substitute(expand(eq), Dict(x^j => 0 for j=last(ns)+1:max_power))

    eqs = []
    for i in ns
        powers = Dict(x^j => (i==j ? 1 : 0) for j=1:last(ns))
        push!(eqs, substitute(eq, powers))
    end
    eqs
end

function solve_coef(eqs, ps)
    vals = Dict()

    for i = 1:length(ps)
        eq = substitute(eqs[i], vals)
        vals[ps[i]] = Symbolics.solve_for(eq ~ 0, ps[i])
    end
    vals
end
```

```
solve_coef (generic function with 1 method)
```





## The Trajectory of a Ball!

In the first two examples, we applied the perturbation method to algebraic problems. However, the main power of the perturbation method is to solve differential equations (usually ODEs, but also occasionally PDEs). Surprisingly, the main procedure developed to solve algebraic problems works well for differential equations. In fact, we will use the same two helper functions, `collect_powers` and `solve_coef`. The main difference is in the way we expand the dependent variables. For algebraic problems, the coefficients of $\epsilon$ are constants; whereas, for differential equations, they are functions of the dependent variable (usually time).

As the first ODE example, we have chosen a simple and well-behaved problem, which is a variation of a standard first-year physics problem: what is the trajectory of an object (say, a ball or a rocket) thrown vertically at velocity $v$ from the surface of a planet? Assuming a constant acceleration of gravity, $g$, every burgeoning physicist knows the answer: $x(t) = x(0) + vt - \frac{1}{2}gt^2$. However, what happens if $g$ is not constant? Specifically, $g$ is inversely proportional to the distant from the center of the planet. If $v$ is large and the projectile travels a large fraction of the radius of the planet, the assumption of constant gravity does not hold anymore. However, unless $v$ is large compared to the escape velocity, the correction is usually small. After simplifications and change of variables to dimensionless, the problem becomes

$$
  \ddot{x}(t) = -\frac{1}{(1 + \epsilon x(t))^2}
  \,,
$$

with the initial conditions $x(0) = 0$, and $\dot{x}(0) = 1$. Note that for $\epsilon = 0$, this equation transforms back to the standard one. Let's start with defining the variables

```julia
n = 2
@variables ϵ t y[0:n](t) ∂∂y[0:n]
```

```
4-element Vector{Any}:
 ϵ
 t
  Symbolics.Num[y₀(t), y₁(t), y₂(t)]
  Symbolics.Num[∂∂y₀, ∂∂y₁, ∂∂y₂]
```





Next, we define $x$.

```julia
x = def_taylor(ϵ, y[2:end], y[1])
```

```
y₀(t) + ϵ*y₁(t) + y₂(t)*(ϵ^2)
```





We need the second derivative of `x`. It may seem that we can do this using `Differential(t)`; however, this operation needs to wait for a few steps because we need to manipulate the differentials as separate variables. Instead, we define dummy variables `∂∂y` as the placeholder for the second derivatives and define

```julia
∂∂x = def_taylor(ϵ, ∂∂y[2:end], ∂∂y[1])
```

```
∂∂y₀ + ϵ*∂∂y₁ + ∂∂y₂*(ϵ^2)
```





as the second derivative of `x`. After rearrangement, our governing equation is $\ddot{x}(t)(1 + \epsilon x(t))^{-2} + 1 = 0$, or

```julia
eq = ∂∂x * (1 + ϵ*x)^2 + 1
```

```
1 + (∂∂y₀ + ϵ*∂∂y₁ + ∂∂y₂*(ϵ^2))*((1 + ϵ*(y₀(t) + ϵ*y₁(t) + y₂(t)*(ϵ^2)))^2
)
```





The next two steps are the same as the ones for algebraic equations (note that we pass `0:n` to `collect_powers` because the zeroth order term is needed here)

```julia
eqs = collect_powers(eq, ϵ, 0:n)
```

```
3-element Vector{Any}:
                                                     1 + ∂∂y₀
                                1 + ∂∂y₀ + ∂∂y₁ + 2∂∂y₀*y₀(t)
 1 + ∂∂y₀ + ∂∂y₂ + ∂∂y₀*(y₀(t)^2) + 2∂∂y₀*y₁(t) + 2∂∂y₁*y₀(t)
```





and,

```julia
vals = solve_coef(eqs, ∂∂y)
```

```
Dict{Any, Any} with 3 entries:
  ∂∂y₁ => 2.0y₀(t)
  ∂∂y₀ => -1.0
  ∂∂y₂ => 2.0y₁(t) - (3.0(y₀(t)^2))
```





Our system of ODEs is forming. Now is the time to convert `∂∂`s to the correct **Symbolics.jl** form by substitution:

```julia
D = Differential(t)
subs = Dict(∂∂y[i] => D(D(y[i])) for i in eachindex(y))
eqs = [substitute(first(v), subs) ~ substitute(last(v), subs) for v in vals]
```

```
3-element Vector{Symbolics.Equation}:
 Differential(t)(Differential(t)(y₁(t))) ~ 2.0y₀(t)
 Differential(t)(Differential(t)(y₀(t))) ~ -1.0
 Differential(t)(Differential(t)(y₂(t))) ~ 2.0y₁(t) - (3.0(y₀(t)^2))
```





We are nearly there! From this point on, the rest is standard ODE solving procedures. Potentially we can use a symbolic ODE solver to find a closed form solution to this problem. However, **Symbolics.jl** currently does not support this functionality. Instead, we solve the problem numerically. We form an `ODESystem`, lower the order (convert second derivatives to first), generate an `ODEProblem` (after passing the correct initial conditions), and, finally, solve it.

```julia
using ModelingToolkit, DifferentialEquations

sys = ODESystem(eqs, t)
sys = ode_order_lowering(sys)
states(sys)
```

```
6-element Vector{Any}:
 y₁ˍt(t)
 y₀ˍt(t)
 y₂ˍt(t)
 y₁(t)
 y₀(t)
 y₂(t)
```



```julia
# the initial conditions
# everything is zero except the initial velocity
u0 = zeros(2n+2)
u0[3] = 1.0   # y₀ˍt

prob = ODEProblem(sys, u0, (0, 3.0))
sol = solve(prob; dtmax=0.01)
```

```
retcode: Success
Interpolation: automatic order switching interpolation
t: 303-element Vector{Float64}:
 0.0
 0.0007064003808057418
 0.007770404188863159
 0.017770404188863158
 0.02777040418886316
 0.03777040418886316
 0.047770404188863164
 0.057770404188863166
 0.06777040418886317
 0.07777040418886316
 ⋮
 2.927770404188845
 2.9377704041888446
 2.9477704041888444
 2.957770404188844
 2.967770404188844
 2.9777704041888438
 2.9877704041888435
 2.9977704041888433
 3.0
u: 303-element Vector{Vector{Float64}}:
 [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
 [-1.174982827371994e-10, -0.0007064003808057417, 1.0, -2.0750207917394702e
-14, -2.4950074900124696e-7, 0.0007064003808057417]
 [-1.5639021432321165e-7, -0.0077704041888631585, 0.9999999999948065, -3.03
8037941185756e-10, -3.0189590629150915e-5, 0.007770404188856389]
 [-1.8705557791258766e-6, -0.017770404188863158, 0.9999999996751162, -8.310
133063220138e-9, -0.00015789363251778217, 0.017770404187900546]
 [-7.138802181701145e-6, -0.027770404188863156, 0.9999999969720239, -4.9561
855502544754e-8, -0.0003855976744064134, 0.027770404174847732]
 [-1.796112942204903e-5, -0.037770404188863155, 0.9999999859071251, -1.6959
97794898182e-7, -0.0007133017162950444, 0.037770404100146475]
 [-3.6337537500169564e-5, -0.04777040418886315, 0.9999999543925265, -4.3396
47134027663e-7, -0.0011410057581836756, 0.047770403825747154]
 [-6.42680264160627e-5, -0.057770404188863145, 0.999999882030846, -9.281974
65619118e-7, -0.0016687098000723068, 0.0577704030530071]
 [-0.00010375259616972848, -0.06777040418886314, 0.9999997379152122, -1.757
838844516597e-6, -0.0022964138419609374, 0.0677704012285957]
 [-0.00015679124676116685, -0.07777040418886313, 0.9999994784292653, -3.048
4296584729253e-6, -0.003024117883849568, 0.07777039742839956]
 ⋮
 [-8.36545937120874, -2.927770404188845, -38.439056472977875, -6.1230360911
17333, -4.285919769822102, -16.31697998093089]
 [-8.451470876978934, -2.9377704041888446, -39.11720718755033, -6.207120253
563204, -4.31524747386399, -16.7047535915276]
 [-8.538069936829967, -2.9477704041888444, -39.804654573946586, -6.29206746
7170514, -4.344675177905878, -17.09935511351616]
 [-8.625258550761837, -2.957770404188844, -40.50149389131581, -6.3778836174
80072, -4.374202881947767, -17.50087798937105]
 [-8.713038718774545, -2.967770404188844, -41.20782104731666, -6.4645746100
32686, -4.403830585989655, -17.90941661739895]
 [-8.80141244086809, -2.9777704041888438, -41.923732600317294, -6.552146370
369165, -4.433558290031543, -18.32506635823484]
 [-8.890381717042473, -2.9877704041888435, -42.649325761595335, -6.64060484
4030317, -4.463385994073432, -18.747923541360098]
 [-8.979948547297694, -2.9977704041888433, -43.38469839753792, -6.729955996
556951, -4.493313698115321, -19.178085471642593]
 [-9.000000000000174, -3.0, -43.55000000000142, -6.750000000000175, -4.5000
00000000051, -19.275000000000954]
```





Finally, we calculate the solution to the problem as a function of `ϵ` by substituting the solution to the ODE system back into the defining equation for `x`. Note that `𝜀` is a number, compared to `ϵ`, which is a symbolic variable.

```julia
X = 𝜀 -> sum([𝜀^(i-1) * sol[y[i]] for i in eachindex(y)])
```

```
#16 (generic function with 1 method)
```





Using `X`, we can plot the trajectory for a range of $𝜀$s.

```julia
using Plots

plot(sol.t, hcat([X(𝜀) for 𝜀 = 0.0:0.1:0.5]...))
```

```
Error: ArgumentError: Package Plots not found in current path:
- Run `import Pkg; Pkg.add("Plots")` to install the Plots package.
```





As expected, as `𝜀` becomes larger (meaning the gravity is less with altitude), the object goes higher and stays up for a longer duration. Of course, we could have solved the problem directly using as ODE solver. One of the benefits of the perturbation method is that we need to run the ODE solver only once and then can just calculate the answer for different values of `𝜀`; whereas, if we had used the direct method, we would need to run the solver once for each value of `𝜀`.

## A Weakly Nonlinear Oscillator

For the next example, we have chosen a simple example from a very important class of problems, the nonlinear oscillators. As we will see, perturbation theory has difficulty providing a good solution to this problem, but the process is instructive. This example closely follows the chapter 7.6 of *Nonlinear Dynamics and Chaos* by Steven Strogatz.

The goal is to solve $\ddot{x} + 2\epsilon\dot{x} + x = 0$, where the dot signifies time-derivatives and the initial conditions are $x(0) = 0$ and $\dot{x}(0) = 1$. If $\epsilon = 0$, the problem reduces to the simple linear harmonic oscillator with the exact solution $x(t) = \sin(t)$. We follow the same steps as the previous example.

```julia
n = 2
@variables ϵ t y[0:n](t) ∂y[0:n] ∂∂y[0:n]
x = def_taylor(ϵ, y[2:end], y[1])
∂x = def_taylor(ϵ, ∂y[2:end], ∂y[1])
∂∂x = def_taylor(ϵ, ∂∂y[2:end], ∂∂y[1])
```

```
∂∂y₀ + ϵ*∂∂y₁ + ∂∂y₂*(ϵ^2)
```





This time we also need the first derivative terms. Continuing,

```julia
eq = ∂∂x + 2*ϵ*∂x + x
eqs = collect_powers(eq, ϵ, 0:n)
vals = solve_coef(eqs, ∂∂y)
```

```
Dict{Any, Any} with 3 entries:
  ∂∂y₁ => -2.0∂y₀ - y₁(t)
  ∂∂y₀ => -y₀(t)
  ∂∂y₂ => -2.0∂y₁ - y₂(t)
```





Next, we need to replace `∂`s and `∂∂`s with their **Symbolics.jl** counterparts:

```julia
D = Differential(t)
subs1 = Dict(∂y[i] => D(y[i]) for i in eachindex(y))
subs2 = Dict(∂∂y[i] => D(D(y[i])) for i in eachindex(y))
subs = subs1 ∪ subs2
eqs = [substitute(first(v), subs) ~ substitute(last(v), subs) for v in vals]
```

```
3-element Vector{Symbolics.Equation}:
 Differential(t)(Differential(t)(y₁(t))) ~ -y₁(t) - (2.0Differential(t)(y₀(
t)))
 Differential(t)(Differential(t)(y₀(t))) ~ -y₀(t)
 Differential(t)(Differential(t)(y₂(t))) ~ -y₂(t) - (2.0Differential(t)(y₁(
t)))
```





We continue with converting 'eqs' to an `ODEProblem`, solving it, and finally plot the results against the exact solution to the original problem, which is $x(t, \epsilon) = (1 - \epsilon)^{-1/2} e^{-\epsilon t} \sin((1- \epsilon^2)^{1/2}t)$,

```julia
sys = ODESystem(eqs, t)
sys = ode_order_lowering(sys)
```

```
Model ##ODESystem#2095 with 6 equations
States (6):
  y₁ˍt(t)
  y₀ˍt(t)
  y₂ˍt(t)
  y₁(t)
⋮
Parameters (0):
```



```julia
# the initial conditions
u0 = zeros(2n+2)
u0[3] = 1.0   # y₀ˍt

prob = ODEProblem(sys, u0, (0, 50.0))
sol = solve(prob; dtmax=0.01)

X = 𝜀 -> sum([𝜀^(i-1) * sol[y[i]] for i in eachindex(y)])
T = sol.t
Y = 𝜀 -> exp.(-𝜀*T) .* sin.(sqrt(1 - 𝜀^2)*T) / sqrt(1 - 𝜀^2)    # exact solution

plot(sol.t, [Y(0.1), X(0.1)])
```

```
Error: UndefVarError: plot not defined
```





The figure is similar to Figure 7.6.2 in *Nonlinear Dynamics and Chaos*. The two curves fit well for the first couple of cycles, but then the perturbation method curve diverges from the true solution. The main reason is that the problem has two or more time-scales that introduce secular terms in the solution. One solution is to explicitly account for the two time scales and use an analytic method called *two-timing*.


## Appendix

These tutorials are a part of the SciMLTutorials.jl repository, found at: [https://github.com/SciML/SciMLTutorials.jl](https://github.com/SciML/SciMLTutorials.jl). For more information on high-performance scientific machine learning, check out the SciML Open Source Software Organization [https://sciml.ai](https://sciml.ai).

To locally run this tutorial, do the following commands:

```
using SciMLTutorials
SciMLTutorials.weave_file("tutorials/perturbation","02-perturbation_differential.jmd")
```

Computer Information:

```
Julia Version 1.6.1
Commit 6aaedecc44 (2021-04-23 05:59 UTC)
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
      Status `/var/lib/buildkite-agent/builds/3-amdci4-julia-csail-mit-edu/julialang/scimltutorials/tutorials/perturbation/Project.toml`
  [0c46a032] DifferentialEquations v6.17.1
  [961ee093] ModelingToolkit v5.17.3
  [30cb0354] SciMLTutorials v0.9.0
  [d1185830] SymbolicUtils v0.11.2
  [0c5d862f] Symbolics v0.1.25
```

And the full manifest:

```
      Status `/var/lib/buildkite-agent/builds/3-amdci4-julia-csail-mit-edu/julialang/scimltutorials/tutorials/perturbation/Manifest.toml`
  [c3fe647b] AbstractAlgebra v0.16.0
  [1520ce14] AbstractTrees v0.3.4
  [79e6a3ab] Adapt v3.3.0
  [ec485272] ArnoldiMethod v0.1.0
  [4fba245c] ArrayInterface v3.1.15
  [4c555306] ArrayLayouts v0.7.0
  [aae01518] BandedMatrices v0.16.9
  [764a87c0] BoundaryValueDiffEq v2.7.1
  [fa961155] CEnum v0.4.1
  [00ebfdb7] CSTParser v2.5.0
  [d360d2e6] ChainRulesCore v0.9.44
  [b630d9fa] CheapThreads v0.2.5
  [35d6a980] ColorSchemes v3.12.1
  [3da002f7] ColorTypes v0.11.0
  [5ae59095] Colors v0.12.8
  [861a8166] Combinatorics v1.0.2
  [a80b9123] CommonMark v0.8.1
  [38540f10] CommonSolve v0.2.0
  [bbf7d656] CommonSubexpressions v0.3.0
  [34da2185] Compat v3.30.0
  [8f4d0f93] Conda v1.5.2
  [187b0558] ConstructionBase v1.2.1
  [d38c429a] Contour v0.5.7
  [a8cc5b0e] Crayons v4.0.4
  [9a962f9c] DataAPI v1.6.0
  [864edb3b] DataStructures v0.18.9
  [e2d170a0] DataValueInterfaces v1.0.0
  [bcd4f6db] DelayDiffEq v5.31.0
  [2b5f629d] DiffEqBase v6.62.2
  [459566f4] DiffEqCallbacks v2.16.1
  [5a0ffddc] DiffEqFinancial v2.4.0
  [c894b116] DiffEqJump v6.14.2
  [77a26b50] DiffEqNoiseProcess v5.7.3
  [055956cb] DiffEqPhysics v3.9.0
  [163ba53b] DiffResults v1.0.3
  [b552c78f] DiffRules v1.0.2
  [0c46a032] DifferentialEquations v6.17.1
  [c619ae07] DimensionalPlotRecipes v1.2.0
  [b4f34e82] Distances v0.10.3
  [31c24e10] Distributions v0.24.18
  [ffbed154] DocStringExtensions v0.8.4
  [e30172f5] Documenter v0.26.3
  [d4d017d3] ExponentialUtilities v1.8.4
  [e2ba6199] ExprTools v0.1.3
  [c87230d0] FFMPEG v0.4.0
  [7034ab61] FastBroadcast v0.1.8
  [9aa1b823] FastClosures v0.3.2
  [1a297f60] FillArrays v0.11.7
  [6a86dc24] FiniteDiff v2.8.0
  [53c48c17] FixedPointNumbers v0.8.4
  [59287772] Formatting v0.4.2
  [f6369f11] ForwardDiff v0.10.18
  [069b7b12] FunctionWrappers v1.1.2
  [28b8d3ca] GR v0.57.4
  [5c1252a2] GeometryBasics v0.3.12
  [42e2da0e] Grisu v1.0.2
  [cd3eb016] HTTP v0.9.9
  [eafb193a] Highlights v0.4.5
  [0e44f5e4] Hwloc v2.0.0
  [7073ff75] IJulia v1.23.2
  [b5f81e59] IOCapture v0.1.1
  [615f187c] IfElse v0.1.0
  [d25df0c9] Inflate v0.1.2
  [83e8ac13] IniFile v0.5.0
  [c8e1da08] IterTools v1.3.0
  [42fd0dbc] IterativeSolvers v0.9.1
  [82899510] IteratorInterfaceExtensions v1.0.0
  [692b3bcd] JLLWrappers v1.3.0
  [682c06a0] JSON v0.21.1
  [98e50ef6] JuliaFormatter v0.13.7
  [b964fa9f] LaTeXStrings v1.2.1
  [2ee39098] LabelledArrays v1.6.1
  [23fbe1c1] Latexify v0.15.5
  [093fc24a] LightGraphs v1.3.5
  [d3d80556] LineSearches v7.1.1
  [2ab3a3ac] LogExpFunctions v0.2.4
  [bdcacae8] LoopVectorization v0.12.23
  [1914dd2f] MacroTools v0.5.6
  [739be429] MbedTLS v1.0.3
  [442fdcdd] Measures v0.3.1
  [e1d29d7a] Missings v1.0.0
  [961ee093] ModelingToolkit v5.17.3
  [46d2c3a1] MuladdMacro v0.2.2
  [f9640e96] MultiScaleArrays v1.8.1
  [ffc61752] Mustache v1.0.10
  [d41bc354] NLSolversBase v7.8.0
  [2774e3e8] NLsolve v4.5.1
  [77ba4419] NaNMath v0.3.5
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
  [1fd47b50] QuadGK v2.4.1
  [74087812] Random123 v1.3.1
  [fb686558] RandomExtensions v0.4.3
  [e6cf234a] RandomNumbers v1.4.0
  [3cdcf5f2] RecipesBase v1.1.1
  [01d81517] RecipesPipeline v0.3.2
  [731186ca] RecursiveArrayTools v2.11.4
  [f2c3362d] RecursiveFactorization v0.1.12
  [189a3867] Reexport v1.0.0
  [ae029012] Requires v1.1.3
  [ae5879a3] ResettableStacks v1.1.0
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
  [b85f4697] SoftGlobalScope v1.1.0
  [a2af1166] SortingAlgorithms v1.0.0
  [47a9eef4] SparseDiffTools v1.13.2
  [276daf66] SpecialFunctions v1.4.1
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
  [8290d209] ThreadingUtilities v0.4.4
  [a759f4b9] TimerOutputs v0.5.9
  [0796e94c] Tokenize v0.5.16
  [a2a6695c] TreeViews v0.3.0
  [5c2747f8] URIs v1.3.0
  [3a884ed6] UnPack v1.0.2
  [1986cc42] Unitful v1.7.0
  [3d5dd08c] VectorizationBase v0.20.11
  [81def892] VersionParsing v1.2.0
  [19fa3120] VertexSafeGraphs v0.1.2
  [44d3d7a6] Weave v0.10.8
  [ddb6d928] YAML v0.4.6
  [c2297ded] ZMQ v1.2.1
  [700de1a5] ZygoteRules v0.2.1
  [6e34b625] Bzip2_jll v1.0.6+5
  [83423d85] Cairo_jll v1.16.0+6
  [5ae413db] EarCut_jll v2.1.5+1
  [2e619515] Expat_jll v2.2.10+0
  [b22a6f82] FFMPEG_jll v4.3.1+4
  [a3f928ae] Fontconfig_jll v2.13.1+14
  [d7e528f0] FreeType2_jll v2.10.1+5
  [559328eb] FriBidi_jll v1.0.5+6
  [0656b61e] GLFW_jll v3.3.4+0
  [d2c73de3] GR_jll v0.57.2+0
  [78b55507] Gettext_jll v0.21.0+0
  [7746bdde] Glib_jll v2.68.1+0
  [e33a78d0] Hwloc_jll v2.4.1+0
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

