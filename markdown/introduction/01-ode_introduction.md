---
author: "Chris Rackauckas"
title: "An Intro to DifferentialEquations.jl"
---





## Basic Introduction Via Ordinary Differential Equations

This notebook will get you started with DifferentialEquations.jl by introducing you to the functionality for solving ordinary differential equations (ODEs). The corresponding documentation page is the [ODE tutorial](https://docs.sciml.ai/dev/tutorials/ode_example/). While some of the syntax may be different for other types of equations, the same general principles hold in each case. Our goal is to give a gentle and thorough introduction that highlights these principles in a way that will help you generalize what you have learned.

### Background

If you are new to the study of differential equations, it can be helpful to do a quick background read on [the definition of ordinary differential equations](https://en.wikipedia.org/wiki/Ordinary_differential_equation). We define an ordinary differential equation as an equation which describes the way that a variable $u$ changes, that is

$$u' = f(u,p,t)$$

where $p$ are the parameters of the model, $t$ is the time variable, and $f$ is the nonlinear model of how $u$ changes. The initial value problem also includes the information about the starting value:

$$u(t_0) = u_0$$

Together, if you know the starting value and you know how the value will change with time, then you know what the value will be at any time point in the future. This is the intuitive definition of a differential equation.

### First Model: Exponential Growth

Our first model will be the canonical exponential growth model. This model says that the rate of change is proportional to the current value, and is this:

$$u' = au$$

where we have a starting value $u(0)=u_0$. Let's say we put 1 dollar into Bitcoin which is increasing at a rate of $98\%$ per year. Then calling now $t=0$ and measuring time in years, our model is:

$$u' = 0.98u$$

and $u(0) = 1.0$. We encode this into Julia by noticing that, in this setup, we match the general form when

```julia
f(u,p,t) = 0.98u
```

```
f (generic function with 1 method)
```





with $u_0 = 1.0$. If we want to solve this model on a time span from `t=0.0` to `t=1.0`, then we define an `ODEProblem` by specifying this function `f`, this initial condition `u0`, and this time span as follows:

```julia
using DifferentialEquations
f(u,p,t) = 0.98u
u0 = 1.0
tspan = (0.0,1.0)
prob = ODEProblem(f,u0,tspan)
```

```
ODEProblem with uType Float64 and tType Float64. In-place: false
timespan: (0.0, 1.0)
u0: 1.0
```





To solve our `ODEProblem` we use the command `solve`.

```julia
sol = solve(prob)
```

```
retcode: Success
Interpolation: automatic order switching interpolation
t: 5-element Vector{Float64}:
 0.0
 0.10042494449239292
 0.35218603951893646
 0.6934436028208104
 1.0
u: 5-element Vector{Float64}:
 1.0
 1.1034222047865465
 1.4121908848175448
 1.9730384275622996
 2.664456142481451
```





and that's it: we have succesfully solved our first ODE!

#### Analyzing the Solution

Of course, the solution type is not interesting in and of itself. We want to understand the solution! The documentation page which explains in detail the functions for analyzing the solution is the [Solution Handling](https://docs.sciml.ai/dev/basics/solution/) page. Here we will describe some of the basics. You can plot the solution using the plot recipe provided by [Plots.jl](http://docs.juliaplots.org/dev/):

```julia
using Plots; gr()
plot(sol)
```

![](figures/01-ode_introduction_4_1.png)



From the picture we see that the solution is an exponential curve, which matches our intuition. As a plot recipe, we can annotate the result using any of the [Plots.jl attributes](http://docs.juliaplots.org/dev/attributes/). For example:

```julia
plot(sol,linewidth=5,title="Solution to the linear ODE with a thick line",
     xaxis="Time (t)",yaxis="u(t) (in μm)",label="My Thick Line!") # legend=false
```

![](figures/01-ode_introduction_5_1.png)



Using the mutating `plot!` command we can add other pieces to our plot. For this ODE we know that the true solution is $u(t) = u_0 exp(at)$, so let's add some of the true solution to our plot:

```julia
plot!(sol.t, t->1.0*exp(0.98t),lw=3,ls=:dash,label="True Solution!")
```

![](figures/01-ode_introduction_6_1.png)



In the previous command I demonstrated `sol.t`, which grabs the array of time points that the solution was saved at:

```julia
sol.t
```

```
5-element Vector{Float64}:
 0.0
 0.10042494449239292
 0.35218603951893646
 0.6934436028208104
 1.0
```





We can get the array of solution values using `sol.u`:

```julia
sol.u
```

```
5-element Vector{Float64}:
 1.0
 1.1034222047865465
 1.4121908848175448
 1.9730384275622996
 2.664456142481451
```





`sol.u[i]` is the value of the solution at time `sol.t[i]`. We can compute arrays of functions of the solution values using standard comprehensions, like:

```julia
[t+u for (u,t) in tuples(sol)]
```

```
5-element Vector{Float64}:
 1.0
 1.2038471492789395
 1.7643769243364813
 2.66648203038311
 3.664456142481451
```





However, one interesting feature is that, by default, the solution is a continuous function. If we check the print out again:

```julia
sol
```

```
retcode: Success
Interpolation: automatic order switching interpolation
t: 5-element Vector{Float64}:
 0.0
 0.10042494449239292
 0.35218603951893646
 0.6934436028208104
 1.0
u: 5-element Vector{Float64}:
 1.0
 1.1034222047865465
 1.4121908848175448
 1.9730384275622996
 2.664456142481451
```





you see that it says that the solution has a order changing interpolation. The default algorithm automatically switches between methods in order to handle all types of problems. For non-stiff equations (like the one we are solving), it is a continuous function of 4th order accuracy. We can call the solution as a function of time `sol(t)`. For example, to get the value at `t=0.45`, we can use the command:

```julia
sol(0.45)
```

```
1.554261048055312
```





#### Controlling the Solver

DifferentialEquations.jl has a common set of solver controls among its algorithms which can be found [at the Common Solver Options](https://diffeq.sciml.ai/stable/basics/common_solver_opts/) page. We will detail some of the most widely used options.

The most useful options are the tolerances `abstol` and `reltol`. These tell the internal adaptive time stepping engine how precise of a solution you want. Generally, `reltol` is the relative accuracy while `abstol` is the accuracy when `u` is near zero. These tolerances are local tolerances and thus are not global guarantees. However, a good rule of thumb is that the total solution accuracy is 1-2 digits less than the relative tolerances. Thus for the defaults `abstol=1e-6` and `reltol=1e-3`, you can expect a global accuracy of about 1-2 digits. If we want to get around 6 digits of accuracy, we can use the commands:

```julia
sol = solve(prob,abstol=1e-8,reltol=1e-8)
```

```
retcode: Success
Interpolation: automatic order switching interpolation
t: 9-element Vector{Float64}:
 0.0
 0.04127492324135852
 0.14679917846877366
 0.28631546412766684
 0.4381941361169628
 0.6118924302028597
 0.7985659100883337
 0.9993516479536952
 1.0
u: 9-element Vector{Float64}:
 1.0
 1.0412786454705882
 1.1547261252949712
 1.3239095703537043
 1.5363819257509728
 1.8214895157178692
 2.1871396448296223
 2.662763824115295
 2.664456241933517
```





Now we can see no visible difference against the true solution:


```julia
plot(sol)
plot!(sol.t, t->1.0*exp(0.98t),lw=3,ls=:dash,label="True Solution!")
```

![](figures/01-ode_introduction_13_1.png)



Notice that by decreasing the tolerance, the number of steps the solver had to take was `9` instead of the previous `5`. There is a trade off between accuracy and speed, and it is up to you to determine what is the right balance for your problem.

Another common option is to use `saveat` to make the solver save at specific time points. For example, if we want the solution at an even grid of `t=0.1k` for integers `k`, we would use the command:

```julia
sol = solve(prob,saveat=0.1)
```

```
retcode: Success
Interpolation: 1st order linear
t: 11-element Vector{Float64}:
 0.0
 0.1
 0.2
 0.3
 0.4
 0.5
 0.6
 0.7
 0.8
 0.9
 1.0
u: 11-element Vector{Float64}:
 1.0
 1.102962785129292
 1.2165269512238264
 1.341783821227542
 1.4799379510586077
 1.632316207054161
 1.8003833264983584
 1.9857565541588758
 2.1902158127997695
 2.415725742084496
 2.664456142481451
```





Notice that when `saveat` is used the continuous output variables are no longer saved and thus `sol(t)`, the interpolation, is only first order. We can save at an uneven grid of points by passing a collection of values to `saveat`. For example:

```julia
sol = solve(prob,saveat=[0.2,0.7,0.9])
```

```
retcode: Success
Interpolation: 1st order linear
t: 3-element Vector{Float64}:
 0.2
 0.7
 0.9
u: 3-element Vector{Float64}:
 1.2165269512238264
 1.9857565541588758
 2.415725742084496
```





If we need to reduce the amount of saving, we can also turn off the continuous output directly via `dense=false`:

```julia
sol = solve(prob,dense=false)
```

```
retcode: Success
Interpolation: 1st order linear
t: 5-element Vector{Float64}:
 0.0
 0.10042494449239292
 0.35218603951893646
 0.6934436028208104
 1.0
u: 5-element Vector{Float64}:
 1.0
 1.1034222047865465
 1.4121908848175448
 1.9730384275622996
 2.664456142481451
```





and to turn off all intermediate saving we can use `save_everystep=false`:

```julia
sol = solve(prob,save_everystep=false)
```

```
retcode: Success
Interpolation: 1st order linear
t: 2-element Vector{Float64}:
 0.0
 1.0
u: 2-element Vector{Float64}:
 1.0
 2.664456142481451
```





If we want to solve and only save the final value, we can even set `save_start=false`.

```julia
sol = solve(prob,save_everystep=false,save_start = false)
```

```
retcode: Success
Interpolation: 1st order linear
t: 1-element Vector{Float64}:
 1.0
u: 1-element Vector{Float64}:
 2.664456142481451
```





Note that similarly on the other side there is `save_end=false`.

More advanced saving behaviors, such as saving functionals of the solution, are handled via the `SavingCallback` in the [Callback Library](https://docs.sciml.ai/dev/features/callback_library/#saving_callback-1) which will be addressed later in the tutorial.

#### Choosing Solver Algorithms

There is no best algorithm for numerically solving a differential equation. When you call `solve(prob)`, DifferentialEquations.jl makes a guess at a good algorithm for your problem, given the properties that you ask for (the tolerances, the saving information, etc.). However, in many cases you may want more direct control. A later notebook will help introduce the various *algorithms* in DifferentialEquations.jl, but for now let's introduce the *syntax*.

The most crucial determining factor in choosing a numerical method is the stiffness of the model. Stiffness is roughly characterized by a Jacobian `f` with large eigenvalues. That's quite mathematical, and we can think of it more intuitively: if you have big numbers in `f` (like parameters of order `1e5`), then it's probably stiff. Or, as the creator of the MATLAB ODE Suite, Lawrence Shampine, likes to define it, if the standard algorithms are slow, then it's stiff. We will go into more depth about diagnosing stiffness in a later tutorial, but for now note that if you believe your model may be stiff, you can hint this to the algorithm chooser via `alg_hints = [:stiff]`.

```julia
sol = solve(prob,alg_hints=[:stiff])
```

```
retcode: Success
Interpolation: specialized 3rd order "free" stiffness-aware interpolation
t: 8-element Vector{Float64}:
 0.0
 0.05653299582822294
 0.17270731152826024
 0.3164602871490142
 0.5057500163821153
 0.7292241858994543
 0.9912975001018789
 1.0
u: 8-element Vector{Float64}:
 1.0
 1.0569657840332976
 1.1844199383303913
 1.3636037723365293
 1.6415399686182572
 2.043449143475479
 2.6418256160577602
 2.6644526430553808
```





Stiff algorithms have to solve implicit equations and linear systems at each step so they should only be used when required.

If we want to choose an algorithm directly, you can pass the algorithm type after the problem as `solve(prob,alg)`. For example, let's solve this problem using the `Tsit5()` algorithm, and just for show let's change the relative tolerance to `1e-6` at the same time:

```julia
sol = solve(prob,Tsit5(),reltol=1e-6)
```

```
retcode: Success
Interpolation: specialized 4th order "free" interpolation
t: 10-element Vector{Float64}:
 0.0
 0.028970819746309166
 0.10049147151547619
 0.19458908698515082
 0.3071725081673423
 0.43945421453622546
 0.5883434923759523
 0.7524873357619015
 0.9293021330536031
 1.0
u: 10-element Vector{Float64}:
 1.0
 1.0287982807225062
 1.1034941463604806
 1.2100931078233779
 1.351248605624241
 1.538280340326815
 1.7799346012651116
 2.090571742234628
 2.486102171447025
 2.6644562434913377
```





### Systems of ODEs: The Lorenz Equation

Now let's move to a system of ODEs. The [Lorenz equation](https://en.wikipedia.org/wiki/Lorenz_system) is the famous "butterfly attractor" that spawned chaos theory. It is defined by the system of ODEs:

$$
\begin{align}
\frac{dx}{dt} &= \sigma (y - x)\\
\frac{dy}{dt} &= x (\rho - z) -y\\
\frac{dz}{dt} &= xy - \beta z
\end{align}
$$

To define a system of differential equations in DifferentialEquations.jl, we define our `f` as a vector function with a vector initial condition. Thus, for the vector `u = [x,y,z]'`, we have the derivative function:

```julia
function lorenz!(du,u,p,t)
    σ,ρ,β = p
    du[1] = σ*(u[2]-u[1])
    du[2] = u[1]*(ρ-u[3]) - u[2]
    du[3] = u[1]*u[2] - β*u[3]
end
```

```
lorenz! (generic function with 1 method)
```





Notice here we used the in-place format which writes the output to the preallocated vector `du`. For systems of equations the in-place format is faster. We use the initial condition $u_0 = [1.0,0.0,0.0]$ as follows:

```julia
u0 = [1.0,0.0,0.0]
```

```
3-element Vector{Float64}:
 1.0
 0.0
 0.0
```





Lastly, for this model we made use of the parameters `p`. We need to set this value in the `ODEProblem` as well. For our model we want to solve using the parameters $\sigma = 10$, $\rho = 28$, and $\beta = 8/3$, and thus we build the parameter collection:

```julia
p = (10,28,8/3) # we could also make this an array, or any other type!
```

```
(10, 28, 2.6666666666666665)
```





Now we generate the `ODEProblem` type. In this case, since we have parameters, we add the parameter values to the end of the constructor call. Let's solve this on a time span of `t=0` to `t=100`:

```julia
tspan = (0.0,100.0)
prob = ODEProblem(lorenz!,u0,tspan,p)
```

```
ODEProblem with uType Vector{Float64} and tType Float64. In-place: true
timespan: (0.0, 100.0)
u0: 3-element Vector{Float64}:
 1.0
 0.0
 0.0
```





Now, just as before, we solve the problem:

```julia
sol = solve(prob)
```

```
retcode: Success
Interpolation: automatic order switching interpolation
t: 1300-element Vector{Float64}:
   0.0
   3.5678604836301404e-5
   0.0003924646531993154
   0.003262408518896374
   0.009058077168386882
   0.01695647153663815
   0.027689960628879868
   0.041856351821061455
   0.06024041060823337
   0.08368540639551347
   ⋮
  99.25227436435598
  99.34990050231407
  99.47329750836677
  99.56888278883171
  99.68067056500875
  99.7698930548574
  99.86396908592394
  99.9520070464327
 100.0
u: 1300-element Vector{Vector{Float64}}:
 [1.0, 0.0, 0.0]
 [0.9996434557625105, 0.0009988049817849058, 1.781434788799189e-8]
 [0.9961045497425811, 0.010965399721242457, 2.1469553658389193e-6]
 [0.9693591566959717, 0.089770627357676, 0.0001438019004555601]
 [0.9242043510496474, 0.24228916412927515, 0.0010461626692665619]
 [0.8800455755115648, 0.43873651254178225, 0.003424260317913913]
 [0.8483309815012585, 0.6915629798778471, 0.008487625758932924]
 [0.8495036692770451, 1.0145426674126548, 0.018212090760571238]
 [0.9139069519040545, 1.4425599553295452, 0.036693820689070726]
 [1.088863767524423, 2.052326420986981, 0.07402572431671739]
 ⋮
 [5.791198832787258, 3.8198742723079415, 26.666685868569655]
 [4.959577668148714, 5.1627464679185815, 22.495413403699086]
 [6.653100232410356, 9.122429705812463, 20.41123180056622]
 [9.593094878342031, 12.952212134722183, 23.567760855989487]
 [11.932162424256417, 11.761476358727544, 31.642393421283604]
 [9.907977700774161, 6.112102601076882, 32.83064925680822]
 [6.4119417247899015, 3.408711533396652, 28.483033292250997]
 [4.734282403550539, 3.913681157796059, 24.002168300285124]
 [4.596259899368738, 4.819051128437629, 22.02318896633189]
```





The same solution handling features apply to this case. Thus `sol.t` stores the time points and `sol.u` is an array storing the solution at the corresponding time points.

However, there are a few extra features which are good to know when dealing with systems of equations. First of all, `sol` also acts like an array. `sol[i]` returns the solution at the `i`th time point.

```julia
sol.t[10],sol[10]
```

```
(0.08368540639551347, [1.088863767524423, 2.052326420986981, 0.074025724316
71739])
```





Additionally, the solution acts like a matrix where `sol[j,i]` is the value of the `j`th variable at time `i`:

```julia
sol[2,10]
```

```
2.052326420986981
```





We can get a real matrix by performing a conversion:

```julia
A = Array(sol)
```

```
3×1300 Matrix{Float64}:
 1.0  0.999643     0.996105    0.969359     …   6.41194   4.73428   4.59626
 0.0  0.000998805  0.0109654   0.0897706        3.40871   3.91368   4.81905
 0.0  1.78143e-8   2.14696e-6  0.000143802     28.483    24.0022   22.0232
```





This is the same as sol, i.e. `sol[i,j] = A[i,j]`, but now it's a true matrix. Plotting will by default show the time series for each variable:

```julia
plot(sol)
```

![](figures/01-ode_introduction_29_1.png)



If we instead want to plot values against each other, we can use the `vars` command. Let's plot variable `1` against variable `2` against variable `3`:

```julia
plot(sol,vars=(1,2,3))
```

![](figures/01-ode_introduction_30_1.png)



This is the classic Lorenz attractor plot, where the `x` axis is `u[1]`, the `y` axis is `u[2]`, and the `z` axis is `u[3]`. Note that the plot recipe by default uses the interpolation, but we can turn this off:

```julia
plot(sol,vars=(1,2,3),denseplot=false)
```

![](figures/01-ode_introduction_31_1.png)



Yikes! This shows how calculating the continuous solution has saved a lot of computational effort by computing only a sparse solution and filling in the values! Note that in vars, `0=time`, and thus we can plot the time series of a single component like:

```julia
plot(sol,vars=(0,2))
```

![](figures/01-ode_introduction_32_1.png)



## Internal Types

The last basic user-interface feature to explore is the choice of types. DifferentialEquations.jl respects your input types to determine the internal types that are used. Thus since in the previous cases, when we used `Float64` values for the initial condition, this meant that the internal values would be solved using `Float64`. We made sure that time was specified via `Float64` values, meaning that time steps would utilize 64-bit floats as well. But, by simply changing these types we can change what is used internally.

As a quick example, let's say we want to solve an ODE defined by a matrix. To do this, we can simply use a matrix as input.

```julia
A  = [1. 0  0 -5
      4 -2  4 -3
     -4  0  0  1
      5 -2  2  3]
u0 = rand(4,2)
tspan = (0.0,1.0)
f(u,p,t) = A*u
prob = ODEProblem(f,u0,tspan)
sol = solve(prob)
```

```
retcode: Success
Interpolation: automatic order switching interpolation
t: 10-element Vector{Float64}:
 0.0
 0.0331010730691975
 0.09782085679194133
 0.18191187446836993
 0.28632603546736024
 0.4040506177182259
 0.5444845160456964
 0.6919976340048595
 0.8531307370404774
 1.0
u: 10-element Vector{Matrix{Float64}}:
 [0.014747968966241753 0.14496017235002934; 0.5405146764582902 0.7507124983
144184; 0.8637771912953807 0.24940709712390485; 0.25268820618439447 0.85571
3156943483]
 [-0.031220752042396306 -0.0002374973693238247; 0.5895706606523794 0.659218
6004376548; 0.8739108240812645 0.2691710214579143; 0.29888519570381655 0.92
67801144223753]
 [-0.14591301277137186 -0.3292274747166891; 0.6522286454994926 0.4364608585
4574035; 0.9179976878204164 0.37432482851661675; 0.3713293434375492 1.03537
35570911555]
 [-0.33466916118168755 -0.830561885827794; 0.6841503488204593 0.09511787169
524738; 1.0315302291748207 0.6579870119771181; 0.4228995681616543 1.1050958
393259087]
 [-0.6040917796504578 -1.5242351889406898; 0.6897988920987816 -0.3217920618
0756564; 1.2710081554063146 1.2623307080718313; 0.4059226449899241 1.055484
7932482665]
 [-0.8962221865295689 -2.30368962364882; 0.7337218290317526 -0.621227822492
1441; 1.6668943459910721 2.276948291040312; 0.26154699125208886 0.784176611
7237723]
 [-1.1087487522394643 -3.021989161941192; 0.9825502421303286 -0.46092512671
40586; 2.2542928821824058 3.8578317871619996; -0.09907689404210246 0.120165
06439738461]
 [-0.9909122944702802 -3.194010670474809; 1.6430287316886858 0.657082066660
454; 2.8403269138214866 5.671558778436573; -0.6882495897088279 -0.992685186
5461248]
 [-0.22328081905826624 -2.2155914075850034; 2.9583231888949753 3.2978880867
294094; 3.098548954921636 7.207366231070659; -1.5062144186355564 -2.6275138
367784754]
 [1.2330802577873936 0.15913595025588334; 4.653356702242153 7.0210638225949
7; 2.561691558505567 7.38017123367908; -2.2716399256807716 -4.3116415394514
07]
```





There is no real difference from what we did before, but now in this case `u0` is a `4x2` matrix. Because of that, the solution at each time point is matrix:

```julia
sol[3]
```

```
4×2 Matrix{Float64}:
 -0.145913  -0.329227
  0.652229   0.436461
  0.917998   0.374325
  0.371329   1.03537
```





In DifferentialEquations.jl, you can use any type that defines `+`, `-`, `*`, `/`, and has an appropriate `norm`. For example, if we want arbitrary precision floating point numbers, we can change the input to be a matrix of `BigFloat`:

```julia
big_u0 = big.(u0)
```

```
4×2 Matrix{BigFloat}:
 0.014748  0.14496
 0.540515  0.750712
 0.863777  0.249407
 0.252688  0.855713
```





and we can solve the `ODEProblem` with arbitrary precision numbers by using that initial condition:

```julia
prob = ODEProblem(f,big_u0,tspan)
sol = solve(prob)
```

```
retcode: Success
Interpolation: automatic order switching interpolation
t: 6-element Vector{Float64}:
 0.0
 0.0331010730691975
 0.20696173808837195
 0.489610659955686
 0.8019632243011008
 1.0
u: 6-element Vector{Matrix{BigFloat}}:
 [0.014747968966241753463464192464016377925872802734375 0.14496017235002933
70960889660636894404888153076171875; 0.540514676458290210803170339204370975
494384765625 0.7507124983144184415806421384331770241260528564453125; 0.8637
77191295380664115555191528983414173126220703125 0.2494070971239048528644843
827350996434688568115234375; 0.25268820618439447045489032461773604154586791
9921875 0.8557131569434830442588690857519395649433135986328125]
 [-0.0312207520256040208018462725768585089479916413145051776297731185547387
4685394847 -0.0002374973746370063726334692405823679833621528276691681405138
268086673211980883989; 0.58957066073643911173479246355577664057102835297568
62542100715403525691120468183 0.6592186005224230967284037788575641289056130
819602492754918453048942424903983319; 0.87391082410375363767149139968773694
81770636954336344979263545693218303915793066 0.2691710215066015806169984373
018550540462447164250561309944021007975335095863645; 0.29888519570172968854
00026353766574781912705260650244932924428252198441194030608 0.9267801144152
455228759065993724041412153835396491544923877712258669311108501346]
 [-0.3971455045661057871583007302525178970474339219695392949460070536417855
343798884 -0.99206425850354891442533243302609587203604266704707245494635921
80430391077461121; 0.686839848952166655368657205907536082445026785120917522
0550049890496686824357338 -0.0097469204373119457244449100560149075002122903
68545345560794795264384835036882812; 1.078838985132652140576174768877515276
513955955529273163029467100585355976378404 0.776999990982186926788473874336
5477733242444076038943802849427299417404962980107; 0.4275676511041581592857
022896097132162174197882847451221030847431641637985708651 1.107953532925116
378505172680061890463539782121177345161807547814240893365355827]
 [-1.0530123841255846942003853171740856687312134559035513042523713533179580
12292795 -2.786064876306964010368736937534528187774392950465400962924153533
294024216106311; 0.84966406408060820968780809929000689031102959682556817866
75784732918771195689298 -0.610780879606481381872221321683947349540242287475
5662811964404240107423300030211; 2.0170249116303885826459953083292604021232
98642614218908427738424621856977225997 3.2038682763560962160817603812110507
97717015562873709130238223977448301117565621; 0.066418611800177891739370928
6447197039374200407751229380603067839449702959776234 0.42567317801766546267
52245985949814421363581849618145926718692312748847773209589]
 [-0.5535152413276966547126085146659920232039809590727065922470779278837273
604446314 -2.68976800980470463593686342293804597085081545304736797936258065
190185270801543; 2.47187664215331604569590372567810304948862165789576535496
6984945876586384284602 2.28912903148757393151483885750368277222422667815978
9301716104703793692750138216; 3.0876802215518665221407072525560459686988869
04564556208894167053935718932374585 6.8226195243784239062360100549807994762
75598698637449619014625500346695188144235; -1.23462731922338697469695995284
8443903590012911068330500317396096190182497812944 -2.0706706059455992375544
20586353046465950126151382554281601996400985221921999]
 [1.23309027327793500473858443206594829803763768955384041283355247669337190
4058872 0.15915056441766642611701673223167539774852295882892334194837309412
92630091850624; 4.653370992046395638011709438733940972844823649045083062177
018407321394365591331 7.021091259367930692153293721068725305308389042073779
974009505267325739194211646; 2.56168888971947932524925499414614763454921782
2190027975030408953260572480212338 7.38017565606460977243860994064483565491
9197887840992375938694496266513805436211; -2.271645575569937310333665370481
826834888720636357471941701763257661280941670421 -4.31165414450274677407613
4011103462276693641638349764709925568719460005028893542]
```



```julia
sol[1,3]
```

```
-0.397145504566105787158300730252517897047433921969539294946007053641785534
3798884
```





To really make use of this, we would want to change `abstol` and `reltol` to be small! Notice that the type for "time" is different than the type for the dependent variables, and this can be used to optimize the algorithm via keeping multiple precisions. We can convert time to be arbitrary precision as well by defining our time span with `BigFloat` variables:

```julia
prob = ODEProblem(f,big_u0,big.(tspan))
sol = solve(prob)
```

```
retcode: Success
Interpolation: automatic order switching interpolation
t: 6-element Vector{BigFloat}:
 0.0
 0.033101073069197498219941999568621391581962542315611620290345971364972686
27152473
 0.206961738088371965072040594512409027450272557247879927000771045747165639
7814062
 0.489610659955685951052930989970056149632220960513998216744469802303939864
721736
 0.801963224301100794548260585623504944993207531476574643481856572314590487
6403047
 1.0
u: 6-element Vector{Matrix{BigFloat}}:
 [0.014747968966241753463464192464016377925872802734375 0.14496017235002933
70960889660636894404888153076171875; 0.540514676458290210803170339204370975
494384765625 0.7507124983144184415806421384331770241260528564453125; 0.8637
77191295380664115555191528983414173126220703125 0.2494070971239048528644843
827350996434688568115234375; 0.25268820618439447045489032461773604154586791
9921875 0.8557131569434830442588690857519395649433135986328125]
 [-0.0312207520256040232038387457028117409930652255633009388706657410453921
9854447601 -0.0002374973746370136686637621442949619665299533487146233450740
818779101156243420605; 0.58957066073643911377359491293201947836404859768661
67365986143851153298186977966 0.6592186005224230919688959176136864322393562
446374196932441411254077441658956819; 0.87391082410375363833867602081638979
73283504954481039414382322759408495906473331 0.2691710215066015820776253846
115689814288899198725977273849955688518754877735026; 0.29888519570172969060
1269975876238908599840752599329916612079482344130223110089 0.92678011441524
55260232417252949585449587609769117104632981334014384821932547719]
 [-0.3971455045661057954891918188459454214461377140441721991385930938649555
090549053 -0.99206425850354893589133991688314380065088078097692925167662614
18573811027958551; 0.686839848952166655599988873082666423846574173744605450
3303599863696778457809135 -0.0097469204373119594109339829759998156696465656
52844700476726528064185355332316201; 1.078838985132652147201985781688757222
695889620901757605163107608725159900606579 0.776999990982186943470773007101
978319184919072821135498956397012169769920707775; 0.42756765110415815955181
17257120916662970801854204338613846709667290926777850861 1.1079535329251163
7829824033032216956459004002247439798948200991791458853004144]
 [-1.0530123841255846851312186476194671908271315323551008406457128333178467
84176498 -2.786064876306963978190828771078410793083983827348697499249823653
394474679267319; 0.84966406408060819687102792546253325339244021954526426725
57462571502052635434822 -0.610780879606481392451607653268327800505016043417
596310655685661542417854027764; 2.01702491163038855463213866226325310390547
5959665516120104574229754711066722292 3.20386827635609614032603802567170967
2805235380137030673967796960208838451252158; 0.0664186118001779096215214367
205840262416639135566252152759778827543963927643886 0.425673178017665495570
5985483671799144112844292353560306353008638694281612946641]
 [-0.5535152413276967988123056551342428180587250815737412628107849962553929
360043278 -2.68976800980470483244838389861796173188907459114128690581012230
9719712184450036; 2.4718766421533158175623061474116395920685834523232278424
6089404616011438959795 2.28912903148757346571942385721713333227981412614209
2503777534666149548048349013; 3.0876802215518664970258565621914095176945924
0059080687527529180575680691884406 6.82261952437842368344596539195626649189
6610235755258761873006155301830665408277; -1.234627319223386840335441441128
000121822012314818185845076868174824014715699702 -2.07067060594559896590383
2847894022252686872967989397973085132445754667697015884]
 [1.23309027327793456788947971612362856485509898447006448145513716287830045
8327467 0.15915056441766567264259322535151641146428470974144133284346401746
5468874606167; 4.6533709920463951978321918348852261476520021815585890558147
9201812911492306714 7.02109125936792968427604053880132433093800237741520498
3840043148270836352421911; 2.5616888897194795751884709103661813580881154133
92543831043686447544089540449663 7.3801756560646099441157965186700230200637
63050715237114006013750300789150834864; -2.27164557556993714266052590833812
0511387887531916619541317559377769405784530876 -4.3116541445027463778292015
48821765212521286601495713954941234870679463160904216]
```





Let's end by showing a more complicated use of types. For small arrays, it's usually faster to do operations on static arrays via the package [StaticArrays.jl](https://github.com/JuliaArrays/StaticArrays.jl). The syntax is similar to that of normal arrays, but for these special arrays we utilize the `@SMatrix` macro to indicate we want to create a static array.

```julia
using StaticArrays
A  = @SMatrix [ 1.0  0.0 0.0 -5.0
                4.0 -2.0 4.0 -3.0
               -4.0  0.0 0.0  1.0
                5.0 -2.0 2.0  3.0]
u0 = @SMatrix rand(4,2)
tspan = (0.0,1.0)
f(u,p,t) = A*u
prob = ODEProblem(f,u0,tspan)
sol = solve(prob)
```

```
retcode: Success
Interpolation: automatic order switching interpolation
t: 11-element Vector{Float64}:
 0.0
 0.028497485729204085
 0.09677891087850361
 0.1838629699017056
 0.2792755454678705
 0.3849487087012745
 0.5037834259824925
 0.6535425718374789
 0.8219485208527942
 0.9963187838589457
 1.0
u: 11-element Vector{StaticArrays.SMatrix{4, 2, Float64, 8}}:
 [0.6044706715729704 0.6651681883844265; 0.25696570195459634 0.800391802344
2045; 0.5029536403278085 0.3575093833720393; 0.03794176795515902 0.09872799
410985289]
 [0.6090164207338807 0.6645296194502982; 0.35457436842122125 0.853916279412
2843; 0.4362080107118939 0.2855402087083679; 0.1414591941953183 0.176444624
92944807]
 [0.5589668179776879 0.617139289324531; 0.5008618571017903 0.90974959705322
08; 0.29280087873465277 0.1273279080385313; 0.3871952918505474 0.3583276474
184245]
 [0.36634268206197285 0.46125231125305666; 0.5134472682634394 0.84001531044
81114; 0.17421053662677327 -0.022875618363667027; 0.6825695145311004 0.5726
310798164117]
 [-0.009281357125025314 0.17041174891042277; 0.3292691258680993 0.606597367
2565503; 0.17953773911100426 -0.08288929953020502; 0.9584979147235346 0.768
7986346795259]
 [-0.6078927872865104 -0.2833285077901624; -0.05096132785483232 0.210325411
44468148; 0.41730328857484894 0.026313633816218568; 1.1700472012314482 0.91
59657644107604]
 [-1.4549440823926374 -0.9185322671331277; -0.566177652311823 -0.3049762534
505666; 1.0471161872690764 0.42085045378126773; 1.2388752217176495 0.961003
1099688554]
 [-2.6191532157534585 -1.7920259126501163; -1.0407475360982281 -0.831463713
6681373; 2.4398891539460696 1.3669469970508974; 0.9841484657160131 0.780816
5896257838]
 [-3.671802984762903 -2.6075076971462505; -0.7885898418834298 -0.8586739891
952148; 4.691977368425225 2.9568352561684854; 0.13256776797971892 0.1907674
2240190392]
 [-3.8254590396795334 -2.8220823558456196; 1.0282240305582555 0.23124784284
706623; 7.280705717329159 4.848706337087659; -1.4394260741422142 -0.8942786
8657168]
 [-3.8126523625270052 -2.8157427240759465; 1.0879003580861206 0.26960362245
150576; 7.331568628025925 4.886871737110194; -1.4798387330520493 -0.9221991
731068224]
```



```julia
sol[3]
```

```
4×2 StaticArrays.SMatrix{4, 2, Float64, 8} with indices SOneTo(4)×SOneTo(2)
:
 0.558967  0.617139
 0.500862  0.90975
 0.292801  0.127328
 0.387195  0.358328
```





## Conclusion

These are the basic controls in DifferentialEquations.jl. All equations are defined via a problem type, and the `solve` command is used with an algorithm choice (or the default) to get a solution. Every solution acts the same, like an array `sol[i]` with `sol.t[i]`, and also like a continuous function `sol(t)` with a nice plot command `plot(sol)`. The Common Solver Options can be used to control the solver for any equation type. Lastly, the types used in the numerical solving are determined by the input types, and this can be used to solve with arbitrary precision and add additional optimizations (this can be used to solve via GPUs for example!). While this was shown on ODEs, these techniques generalize to other types of equations as well.


## Appendix

These tutorials are a part of the SciMLTutorials.jl repository, found at: [https://github.com/SciML/SciMLTutorials.jl](https://github.com/SciML/SciMLTutorials.jl). For more information on high-performance scientific machine learning, check out the SciML Open Source Software Organization [https://sciml.ai](https://sciml.ai).

To locally run this tutorial, do the following commands:

```
using SciMLTutorials
SciMLTutorials.weave_file("introduction","01-ode_introduction.jmd")
```

Computer Information:

```
Julia Version 1.6.3
Commit ae8452a9e0 (2021-09-23 17:34 UTC)
Platform Info:
  OS: Linux (x86_64-pc-linux-gnu)
  CPU: AMD EPYC 7502 32-Core Processor
  WORD_SIZE: 64
  LIBM: libopenlibm
  LLVM: libLLVM-11.0.1 (ORCJIT, znver2)
Environment:
  JULIA_CPU_THREADS = 16
  BUILDKITE_PLUGIN_JULIA_CACHE_DIR = /cache/julia-buildkite-plugin
  JULIA_DEPOT_PATH = /cache/julia-buildkite-plugin/depots/a6029d3a-f78b-41ea-bc97-28aa57c6c6ea

```

Package Information:

```
      Status `/cache/build/amdci4-4/julialang/scimltutorials-dot-jl/tutorials/introduction/Project.toml`
  [6e4b80f9] BenchmarkTools v1.0.0
  [0c46a032] DifferentialEquations v6.17.1
  [65888b18] ParameterizedFunctions v5.10.0
  [91a5bcdd] Plots v1.15.2
  [30cb0354] SciMLTutorials v0.9.0
  [90137ffa] StaticArrays v1.2.0
  [c3572dad] Sundials v4.4.3
  [37e2e46d] LinearAlgebra
```

And the full manifest:

```
      Status `/cache/build/amdci4-4/julialang/scimltutorials-dot-jl/tutorials/introduction/Manifest.toml`
  [c3fe647b] AbstractAlgebra v0.16.0
  [1520ce14] AbstractTrees v0.3.4
  [79e6a3ab] Adapt v3.3.0
  [ec485272] ArnoldiMethod v0.1.0
  [4fba245c] ArrayInterface v3.1.15
  [4c555306] ArrayLayouts v0.7.0
  [aae01518] BandedMatrices v0.16.9
  [6e4b80f9] BenchmarkTools v1.0.0
  [764a87c0] BoundaryValueDiffEq v2.7.1
  [fa961155] CEnum v0.4.1
  [d360d2e6] ChainRulesCore v0.9.44
  [b630d9fa] CheapThreads v0.2.5
  [35d6a980] ColorSchemes v3.12.1
  [3da002f7] ColorTypes v0.11.0
  [5ae59095] Colors v0.12.8
  [861a8166] Combinatorics v1.0.2
  [38540f10] CommonSolve v0.2.0
  [bbf7d656] CommonSubexpressions v0.3.0
  [34da2185] Compat v3.30.0
  [8f4d0f93] Conda v1.5.2
  [187b0558] ConstructionBase v1.2.1
  [d38c429a] Contour v0.5.7
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
  [615f187c] IfElse v0.1.0
  [d25df0c9] Inflate v0.1.2
  [83e8ac13] IniFile v0.5.0
  [c8e1da08] IterTools v1.3.0
  [42fd0dbc] IterativeSolvers v0.9.1
  [82899510] IteratorInterfaceExtensions v1.0.0
  [692b3bcd] JLLWrappers v1.3.0
  [682c06a0] JSON v0.21.1
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
  [961ee093] ModelingToolkit v5.16.0
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

