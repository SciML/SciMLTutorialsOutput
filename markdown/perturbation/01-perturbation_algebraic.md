---
author: "Shahriar Iravanian"
title: "Mixed Symbolic/Numerical Methods for Perturbation Theory - Algebraic Equations"
---


## Background

[**Symbolics.jl**](https://github.com/JuliaSymbolics/Symbolics.jl) is a fast and modern Computer Algebra System (CAS) written in the Julia Programming Language. It is an integral part of the [SciML](https://sciml.ai/) ecosystem of differential equation solvers and scientific machine learning packages. While **Symbolics.jl** is primarily designed for modern scientific computing (e.g., auto-differentiation, machine learning), it is a powerful CAS and can also be useful for *classic* scientific computing. One such application is using the *perturbation* theory to solve algebraic and differential equations.

Perturbation methods are a collection of techniques to solve intractable problems that generally don't have a closed solution but depend on a tunable parameter and have closed or easy solutions for some values of the parameter. The main idea is to assume a solution as a power series in the tunable parameter (say $ϵ$), such that $ϵ = 0$ corresponds to an easy solution.

We will discuss the general steps of the perturbation methods to solve algebraic (this tutorial) and differential equations (*Mixed Symbolic/Numerical Methods for Perturbation Theory - Differential Equations*).

The hallmark of the perturbation method is the generation of long and convoluted intermediate equations, which are subjected to algorithmic and mechanical manipulations. Therefore, these problems are well suited for CAS. In fact, CAS softwares have been used to help with the perturbation calculations since the early 1970s.

In this tutorial our goal is to show how to use a mix of symbolic manipulations (**Symbolics.jl**) and numerical methods (**DifferentialEquations.jl**) to solve simple perturbation problems.

## Solving the Quintic

We start with the "hello world!" analog of the perturbation problems, solving the quintic (fifth-order) equations. We want to find a real valued $x$ such that $x^5 + x = 1$. According to the Abel's theorem, a general quintic equation does not have a closed form solution. Of course, we can easily solve this equation numerically; for example, by using the Newton's method. We use the following implementation of the Newton's method:

```julia
using Symbolics, SymbolicUtils

function solve_newton(f, x, x₀; abstol=1e-8, maxiter=50)
    xₙ = Float64(x₀)
    fₙ₊₁ = x - f / Symbolics.derivative(f, x)

    for i = 1:maxiter
        xₙ₊₁ = substitute(fₙ₊₁, Dict(x => xₙ))
        if abs(xₙ₊₁ - xₙ) < abstol
            return xₙ₊₁
        else
            xₙ = xₙ₊₁
        end
    end
    return xₙ₊₁
end
```

```
solve_newton (generic function with 1 method)
```





In this code, `Symbolics.derivative(eq, x)` does exactly what it names implies: it calculates the symbolic derivative of `eq` (a **Symbolics.jl** expression) with respect to `x` (a **Symbolics.jl** variable). We use `Symbolics.substitute(eq, D)` to evaluate the update formula by substituting variables or sub-expressions (defined in a dictionary `D`) in `eq`. It should be noted that `substitute` is the workhorse of our code and will be used multiple times in the rest of these tutorials. `solve_newton` is written with simplicity and clarity, and not performance, in mind but suffices for our purpose.

Let's go back to our quintic. We can define a Symbolics variable as `@variables x` and then solve the equation `solve_newton(x^5 + x - 1, x, 1.0)` (here, `x₀ = 0` is our first guess). The answer is 0.7549. Now, let's see how we can solve the same problem using the perturbation methods.

We introduce a tuning parameter $\epsilon$ into our equation: $x^5 + \epsilon x = 1$. If $\epsilon = 1$, we get our original problem. For $\epsilon = 0$, the problem transforms to an easy one: $x^5 = 1$ which has an exact real solution $x = 1$ (and four complex solutions which we ignore here). We expand $x$ as a power series on $\epsilon$:

$$
  x(\epsilon) = a_0 + a_1 \epsilon + a_2 \epsilon^2 + O(\epsilon^3)
  \,.
$$

$a_0$ is the solution of the easy equation, therefore $a_0 = 1$. Substituting into the original problem,

$$
  (a_0 + a_1 \epsilon + a_2 \epsilon^2)^5 + \epsilon (a_0 + a_1 \epsilon + a_2 \epsilon^2) - 1 = 0
  \,.
$$

Expanding the equations, we get
$$
  \epsilon (1 + 5 a_1) + \epsilon^2 (a_1 + 5 a_2 + 10 a1_2) + 𝑂(\epsilon^3) = 0
  \,.
$$

This equation should hold for each power of $\epsilon$. Therefore,

$$
  1 + 5 a_1 = 0
  \,,
$$

and

$$
  a_1 + 5 a_2 + 10 a_1^2 = 0
  \,.
$$

This system of equations does not initially seem to be linear because of the presence of terms like $10 a_1^2$, but upon closer inspection is found to be in fact linear (this is a feature of the perturbation methods). In addition, the system is in a triangular form, meaning the first equation depends only on $a_1$, the second one on $a_1$ and $a_2$, such that we can replace the result of $a_1$ from the first one into the second equation and remove the non-linear term. We solve the first equation to get $a_1 = -\frac{1}{5}$. Substituting in the second one and solve for $a_2$:

$$
  a_2 = \frac{(-\frac{1}{5} + 10(-(\frac{1}{5})²)}{5}  = -\frac{1}{25}
  \,.
$$

Finally,

$$
  x(\epsilon) = 1 - \frac{\epsilon}{5} - \frac{\epsilon^2}{25} + O(\epsilon^3)
  \,.
$$

Solving the original problem, $x(1) = 0.76$, compared to 0.7548 calculated numerically. We can improve the accuracy by including more terms in the expansion of $x$. However, the calculations, while straightforward, become messy and intractable to do manually very quickly. This is why a CAS is very helpful to solve perturbation problems.

Now, let's see how we can do these calculations in Julia. Let $n$ be the order of the expansion. We start by defining the symbolic variables:

```julia
n = 2
@variables ϵ a[1:n]
```

```
2-element Vector{Any}:
 ϵ
  Symbolics.Num[a₁, a₂]
```





Then, we define

```julia
x = 1 + a[1]*ϵ + a[2]*ϵ^2
```

```
1 + a₁*ϵ + a₂*(ϵ^2)
```





The next step is to substitute `x` in the problem equation

```julia
  eq = x^5 + ϵ*x - 1
```

```
ϵ*(1 + a₁*ϵ + a₂*(ϵ^2)) + (1 + a₁*ϵ + a₂*(ϵ^2))^5 - 1
```





The expanded form of `eq` is

```julia
expand(eq)
```

```
ϵ + a₁*(ϵ^2) + a₂*(ϵ^3) + (a₁^5)*(ϵ^5) + (a₂^5)*(ϵ^10) + 5a₁*ϵ + 5a₂*(ϵ^2) 
+ 10(a₁^2)*(ϵ^2) + 10(a₁^3)*(ϵ^3) + 5(a₁^4)*(ϵ^4) + 10(a₂^2)*(ϵ^4) + 10(a₂^
3)*(ϵ^6) + 5(a₂^4)*(ϵ^8) + 20a₁*a₂*(ϵ^3) + 30a₁*(a₂^2)*(ϵ^5) + 20a₁*(a₂^3)*
(ϵ^7) + 5a₁*(a₂^4)*(ϵ^9) + 30a₂*(a₁^2)*(ϵ^4) + 20a₂*(a₁^3)*(ϵ^5) + 5a₂*(a₁^
4)*(ϵ^6) + 30(a₁^2)*(a₂^2)*(ϵ^6) + 10(a₁^2)*(a₂^3)*(ϵ^8) + 10(a₁^3)*(a₂^2)*
(ϵ^7)
```





We need a way to get the coefficients of different powers of `ϵ`. Function `collect_powers(eq, x, ns)` returns the powers of variable `x` in expression `eq`. Argument `ns` is the range of the powers.

```julia
function collect_powers(eq, x, ns; max_power=100)
    eq = substitute(expand(eq), Dict(x^j => 0 for j=last(ns)+1:max_power))

    eqs = []
    for i in ns
        powers = Dict(x^j => (i==j ? 1 : 0) for j=1:last(ns))
        push!(eqs, substitute(eq, powers))
    end
    eqs
end
```

```
collect_powers (generic function with 1 method)
```





To return the coefficients of $ϵ$ and $ϵ^2$ in `eq`, we can write

```julia
eqs = collect_powers(eq, ϵ, 1:2)
```

```
2-element Vector{Any}:
             1 + 5a₁
 a₁ + 5a₂ + 10(a₁^2)
```





A few words on how `collect_powers` works, It uses `substitute` to find the coefficient of a given power of `x` by passing a `Dict` with all powers of `x` set to 0, except the target power which is set to 1. For example, the following expression returns the coefficient of `ϵ^2` in `eq`,

```julia
substitute(expand(eq), Dict(
  ϵ => 0,
  ϵ^2 => 1,
  ϵ^3 => 0,
  ϵ^4 => 0,
  ϵ^5 => 0,
  ϵ^6 => 0,
  ϵ^7 => 0,
  ϵ^8 => 0)
)
```

```
a₁ + 5a₂ + 10(a₁^2)
```





Back to our problem. Having the coefficients of the powers of `ϵ`, we can set each equation in `eqs` to 0 (remember, we rearrange the problem such that `eq` is 0) and solve the system of linear equations to find the numerical values of the coefficients. **Symbolics.jl** has a function `Symbolics.solve_for` that can solve systems of linear equations. However, the presence of higher order terms in `eqs` prevents `Symbolics.solve_for(eqs .~ 0, a)` from workings properly. Instead, we can exploit the fact that our system is in a triangular form and start by solving `eqs[1]` for `a₁` and then substitute this in `eqs[2]` and solve for `a₂` (as continue the same process for higher order terms).  This *cascading* process is done by function `solve_coef(eqs, ps)`:

```julia
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





Here, `eqs` is an array of expressions (assumed to be equal to 0) and `ps` is an array of variables. The result is a dictionary of *variable* => *value* pairs. We apply `solve_coef` to `eqs` to get the numerical values of the parameters:

```julia
solve_coef(eqs, a)
```

```
Dict{Any, Any} with 2 entries:
  a₂ => -0.04
  a₁ => -0.2
```





Finally, we substitute back the values of `a` in the definition of `x` as a function of `𝜀`. Note that `𝜀` is a number (usually Float64), whereas `ϵ` is a symbolic variable.

```julia
X = 𝜀 -> 1 + a[1]*𝜀 + a[2]*𝜀^2
```

```
#9 (generic function with 1 method)
```





Therefore, the solution to our original problem becomes `X(1)`, which is equal to 0.76. We can use larger values of `n` to improve the accuracy of estimations.

| n | x              |
|---|----------------|
|1  |0.8 |
|2  |0.76|
|3  |0.752|
|4  |0.752|
|5  |0.7533|
|6  |0.7543|
|7  |0.7548|
|8  |0.7550|

Remember the numerical value is 0.7549. The two functions `collect_powers` and `solve_coef(eqs, a)` are used in all the examples in this and the next tutorial.

## Solving the Kepler's Equation

Historically, the perturbation methods were first invented to solve orbital calculations of the Moon and the planets. In homage to this history, our second example has a celestial theme. Our goal is solve the Kepler's equation:

$$
  E - e\sin(E) = M
  \,.
$$

where $e$ is the *eccentricity* of the elliptical orbit, $M$ is the *mean anomaly*, and $E$ (unknown) is the *eccentric anomaly* (the angle between the position of a planet in an elliptical orbit and the point of periapsis). This equation is central to solving two-body Keplerian orbits.

Similar to the first example, it is easy to solve this problem using the Newton's method. For example, let $e = 0.01671$ (the eccentricity of the Earth) and $M = \pi/2$. We have `solve_newton(x - e*sin(x) - M, x, M)` equals to 1.5875 (compared to π/2 = 1.5708). Now, we try to solve the same problem using the perturbation techniques (see function `test_kepler`).

For $e = 0$, we get $E = M$. Therefore, we can use $e$ as our perturbation parameter. For consistency with other problems, we also rename $e$ to $\epsilon$ and $E$ to $x$.

From here on, we use the helper function `def_taylor` to define Taylor's series by calling it as `x = def_taylor(ϵ, a, 1)`, where the arguments are, respectively, the perturbation variable, an array of coefficients (starting from the coefficient of $\epsilon^1$), and an optional constant term.

```julia
def_taylor(x, ps) = sum([a*x^i for (i,a) in enumerate(ps)])
def_taylor(x, ps, p₀) = p₀ + def_taylor(x, ps)
```

```
def_taylor (generic function with 2 methods)
```





We start by defining the variables (assuming `n = 3`):

```julia
n = 3
@variables ϵ M a[1:n]
x = def_taylor(ϵ, a, M)
```

```
M + a₁*ϵ + a₂*(ϵ^2) + a₃*(ϵ^3)
```





We further simplify by substituting `sin` with its power series using the `expand_sin` helper function:

```julia
expand_sin(x, n) = sum([(isodd(k) ? -1 : 1)*(-x)^(2k-1)/factorial(2k-1) for k=1:n])
```

```
expand_sin (generic function with 1 method)
```





To test,

```julia
expand_sin(0.1, 10) ≈ sin(0.1)
```

```
true
```





The problem equation is

```julia
eq = x - ϵ * expand_sin(x, n) - M
```

```
a₁*ϵ + a₂*(ϵ^2) + a₃*(ϵ^3) - (ϵ*(M + a₁*ϵ + a₂*(ϵ^2) + a₃*(ϵ^3) + 0.1666666
6666666666((-M - (a₁*ϵ) - (a₂*(ϵ^2)) - (a₃*(ϵ^3)))^3) - (0.0083333333333333
33((-M - (a₁*ϵ) - (a₂*(ϵ^2)) - (a₃*(ϵ^3)))^5))))
```





We follow the same process as the first example. We collect the coefficients of the powers of `ϵ`

```julia
eqs = collect_powers(eq, ϵ, 1:n)
```

```
3-element Vector{Any}:
 a₁ + 0.16666666666666666(M^3) - M - (0.008333333333333333(M^5))
 a₂ + 0.5a₁*(M^2) - a₁ - (0.041666666666666664a₁*(M^4))
 a₃ + 0.5M*(a₁^2) + 0.5a₂*(M^2) - a₂ - (0.041666666666666664a₂*(M^4)) - (0.
08333333333333333(M^3)*(a₁^2))
```





and then solve for `a`:

```julia
vals = solve_coef(eqs, a)
```

```
Dict{Any, Any} with 3 entries:
  a₂ => M + 0.00833333(M^5) + 0.0416667(M + 0.00833333(M^5) - (0.166667(M^3
)))*…
  a₃ => M + 0.00833333(M^5) + 0.0416667(M + 0.00833333(M^5) - (0.166667(M^3
)))*…
  a₁ => M + 0.00833333(M^5) - (0.166667(M^3))
```





Finally, we substitute `vals` back in `x`:

```julia
x′ = substitute(x, vals)
X = (𝜀, 𝑀) -> substitute(x′, Dict(ϵ => 𝜀, M => 𝑀))
X(0.01671, π/2)
```

```
1.5875853629172587
```





The result is 1.5876, compared to the numerical value of 1.5875. It is customary to order `X` based on the powers of `𝑀` instead of `𝜀`. We can calculate this series as `collect_powers(sol, M, 0:3)
`. The result (after manual cleanup) is

```
(1 + 𝜀 + 𝜀^2 + 𝜀^3)*𝑀
- (𝜀 + 4*𝜀^2 + 10*𝜀^3)*𝑀^3/6
+ (𝜀 + 16*𝜀^2 + 91*𝜀^3)*𝑀^5/120
```

Comparing the formula to the one for 𝐸 in the [Wikipedia article on the Kepler's equation](https://en.wikipedia.org/wiki/Kepler%27s_equation):

$$
  E = \frac{1}{1-\epsilon}M
    -\frac{\epsilon}{(1-\epsilon)^4} \frac{M^3}{3!} + \frac{(9\epsilon^2
    + \epsilon)}{(1-\epsilon)^7}\frac{M^5}{5!}\cdots
$$

The first deviation is in the coefficient of $\epsilon^3 M^5$.


## Appendix

These tutorials are a part of the SciMLTutorials.jl repository, found at: [https://github.com/SciML/SciMLTutorials.jl](https://github.com/SciML/SciMLTutorials.jl). For more information on high-performance scientific machine learning, check out the SciML Open Source Software Organization [https://sciml.ai](https://sciml.ai).

To locally run this tutorial, do the following commands:

```
using SciMLTutorials
SciMLTutorials.weave_file("tutorials/perturbation","01-perturbation_algebraic.jmd")
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
      Status `/var/lib/buildkite-agent/builds/7-amdci4-julia-csail-mit-edu/julialang/scimltutorials-dot-jl/tutorials/perturbation/Project.toml`
  [0c46a032] DifferentialEquations v6.17.1
  [961ee093] ModelingToolkit v5.17.3
  [91a5bcdd] Plots v1.15.2
  [30cb0354] SciMLTutorials v0.9.0
  [d1185830] SymbolicUtils v0.11.2
  [0c5d862f] Symbolics v0.1.25
```

And the full manifest:

```
      Status `/var/lib/buildkite-agent/builds/7-amdci4-julia-csail-mit-edu/julialang/scimltutorials-dot-jl/tutorials/perturbation/Manifest.toml`
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

