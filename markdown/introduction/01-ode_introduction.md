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

DifferentialEquations.jl has a common set of solver controls among its algorithms which can be found [at the Common Solver Options](https://docs.sciml.ai/dev/basics/common_solver_opts/) page. We will detail some of the most widely used options.

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
t: 11-element Vector{Float64}:
 0.0
 0.029308743288177685
 0.08635811823886122
 0.16160775827593726
 0.25537599279844686
 0.3642438244465772
 0.49472696786039494
 0.6458408885231186
 0.8053139197594285
 0.9801750411091223
 1.0
u: 11-element Vector{Matrix{Float64}}:
 [0.9186901247370327 0.05383448668017121; 0.19650898913526493 0.08786333766
46778; 0.8450924397546136 0.16525383719060338; 0.1904812060483081 0.8989444
887017792]
 [0.9036808763305719 -0.08485200211963242; 0.3553845709285253 0.02088571099
608641; 0.7463512041959118 0.19456084319835917; 0.3796642408327901 0.987028
1934916277]
 [0.7922697036895618 -0.40203465134333916; 0.5523843916723207 -0.1526457159
2925175; 0.5828171746467887 0.3096765152565658; 0.742545643071634 1.1361994
164054086]
 [0.4755336119284342 -0.9069034035405004; 0.5931797216771086 -0.44463433298
45744; 0.46027998008720256 0.5956249478402404; 1.1941712903961124 1.2768792
792668782]
 [-0.18593462832748237 -1.6442742718161742; 0.3352333861269305 -0.843350674
8595082; 0.532411051137799 1.1947001754470226; 1.6763226275188554 1.3380974
742980944]
 [-1.2898907087563736 -2.577827321933914; -0.27615223992914734 -1.219536390
1804877; 1.0471546050842413 2.2536448042770303; 2.057027180497904 1.2089235
830507847]
 [-2.96014682897817 -3.632441602786137; -1.1559059024599903 -1.283681693477
134; 2.4234109287609895 4.01198139578686; 2.1446134355966673 0.707491942635
3882]
 [-5.019026890177436 -4.396495732426178; -1.7597769007013322 -0.38756545782
271035; 5.131860554453925 6.505049925873069; 1.5693670628059473 -0.41617973
851693735]
 [-6.6571138702118695 -4.072601847662041; -0.9835963960977889 2.20163269344
1078; 9.047365996960055 9.082604939570999; -0.005759687228249488 -2.2442728
009795054]
 [-6.654316782158411 -1.543365152541004; 2.6897969923534584 7.3289331965807
11; 13.612085646272915 10.592231568498029; -2.955607744175758 -4.8237215283
78857]
 [-6.47135537801696 -1.0758376803993965; 3.3354217340539147 8.0564876087716
1; 14.070181014756319 10.597636407865906; -3.364809333912465 -5.13469509069
6802]
```





There is no real difference from what we did before, but now in this case `u0` is a `4x2` matrix. Because of that, the solution at each time point is matrix:

```julia
sol[3]
```

```
4×2 Matrix{Float64}:
 0.79227   -0.402035
 0.552384  -0.152646
 0.582817   0.309677
 0.742546   1.1362
```





In DifferentialEquations.jl, you can use any type that defines `+`, `-`, `*`, `/`, and has an appropriate `norm`. For example, if we want arbitrary precision floating point numbers, we can change the input to be a matrix of `BigFloat`:

```julia
big_u0 = big.(u0)
```

```
4×2 Matrix{BigFloat}:
 0.91869   0.0538345
 0.196509  0.0878633
 0.845092  0.165254
 0.190481  0.898944
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
 0.029308743288177678
 0.18634126414652574
 0.45349453804173223
 0.7619669363634161
 1.0
u: 6-element Vector{Matrix{BigFloat}}:
 [0.9186901247370327094898811992607079446315765380859375 0.0538344866801712
118586920041707344353199005126953125; 0.19650898913526493316794585552997887
134552001953125 0.0878633376646777985996550341951660811901092529296875; 0.8
450924397546135669045952454325743019580841064453125 0.165253837190603380946
640754700638353824615478515625; 0.19048120604830809377006062277359887957572
93701171875 0.898944488701779231831778815831057727336883544921875]
 [0.90368087632687856577996194053626533914144747264234315732975993300993890
88750249 -0.084852002124718940761081282888566295159360815500189390309394735
93104095542897951; 0.355384571005972908143681525018092724156875673792539167
1930595126668108275608134 0.02088571104589767015250040099407875643099487829
640560284999433487046861832676688; 0.74635120424924845104486728410068084811
71230751893660235302061716794161761520902 0.1945608432366606690392539753757
556713309957304605063531357529969047929455700225; 0.37966424084031877062952
03047141950316452183574343259259768516743754708532135729 0.9870281934856206
796995269127826861066958683115463752344465607567289495327003288]
 [0.32926798584877511130426165408838089870483627407785627656098092870423202
68728642 -1.091431670636088022770644453418388456759407921565102781657579442
040902895218755; 0.55584553466712143640165477302109343154710241958437832875
00688565544638357116009 -0.549499884178480313131765178303343550933582087437
5615450703205548453842113335017; 0.4515508286317397378126744992538513202340
722403689247063315744528443929151554147 0.726375253008073713377508525282641
8496965157047912011804010847714430196296997616; 1.3318665037518214604197707
37399260876888412794959711497130162921318797160760583 1.3064971048448482941
73811781039345909955336921632255614205389681303815285222127]
 [-2.4041501568734571746030507521081461687704720445838066004933254978085677
74932735 -3.321450152020386266842055175759530541391209997257561988023810771
009730190689758; -0.8854614151896058109222781287295007053753588311573342481
364200156209689014725832 -1.32852251348253438937865774199442986161068466140
7354809550731883558098566655349; 1.8922011306553325957739826316690574778657
00435715970330988853288572533811234199 3.4046264751891355279645698922855557
50780581356703301639749137242779887952234741; 2.168344641682897118166068997
599948072785130209024966866901460073353577257833887 0.910586991797430500155
3915526519476336934857418190191918772116664980284168023464]
 [-6.3175018941577904011590657855198281654423275682354343195652914390366426
51336324 -4.315567803957094246531214414984293904664138832470335336089500665
220914052453393; -1.3964296822792695763162421277070374332328618104451173688
41553835425024223916804 1.3066650651428738882016818828453584696077621651586
84147666350414550572955552958; 7.909495573705853300333087405517839798604738
350790959788853986603581576095090558 8.438565247369330045202812991427464281
928006097485397838526091934783778209445346; 0.52807480928563140476866952996
12386266742758759164455596656057488750840620056731 -1.686727457404012213194
387057740599611025553449703058297553216868201714126572271]
 [-6.4713586148365916044986474670942309647414591544866216713999682033230473
35175163 -1.075819223131073816173769931641722870339105748118822599725320220
621103265692855; 3.33545957795870538981707467860140840427294282466657338476
6919989637503723636995 8.05653499295683763257990657268624545625294779964620
5396056679847654558875461246; 14.070225572817499129338766194093885145122888
85351305978040068989970271236956607 10.597655405150589244002764304527880210
1115484298606422880359649706864340256235; -3.364832317612520729293155418229
746716141681791058115420825861372314011400730395 -5.13471798916916942315233
9800117663256584988793179173636251730624463046984599799]
```



```julia
sol[1,3]
```

```
0.3292679858487751113042616540883808987048362740778562765609809287042320268
728642
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
 0.029308743288177678519458622801430102131604699229544997246806259074941912
5262367
 0.186341264146525747712340172152118843072737080655874129261378574031912792
138413
 0.453494538041732218852382790081782792248592736236767760666814136518308507
3489843
 0.761966936363416022115958935198627310816350723811858422789051349046866929
8995236
 1.0
u: 6-element Vector{Matrix{BigFloat}}:
 [0.9186901247370327094898811992607079446315765380859375 0.0538344866801712
118586920041707344353199005126953125; 0.19650898913526493316794585552997887
134552001953125 0.0878633376646777985996550341951660811901092529296875; 0.8
450924397546135669045952454325743019580841064453125 0.165253837190603380946
640754700638353824615478515625; 0.19048120604830809377006062277359887957572
93701171875 0.898944488701779231831778815831057727336883544921875]
 [0.90368087632687856511336636665326333038351786578264752512083094491388888
19497321 -0.084852002124718944125418119680487637093856520178608579181760284
2206584275652019; 0.3553845710059729113273180279125223061547411268691443131
9823801903065967380836 0.02088571104589766843412566712946703679067545105971
704062693347622843700318626521; 0.74635120424924844887677081092544289192228
58380853331099391627876960471754850375 0.1945608432366606699282150163686745
492623453481336786970344210705632159454511144; 0.37966424084031877494507941
03982300496195591054232323001873094970279681931158489 0.9870281934856206816
324650059930892805405634727504891321758238767594779873459864]
 [0.32926798584877509989195312715183474706172638268651672355513641301298936
59443847 -1.091431670636088036515606255752601851817055352540459734504121284
242966459740878; 0.55584553466712143282472520835855728203932513141738425406
33787978380715935464862 -0.549499884178480320849358787836364214730355333197
4721509625078159333499356015237; 0.4515508286317397378393472275789889549394
73493079229309423510722487511231037924 0.7263752530080737236038131011368391
066679611613467819460052748199404332955274411; 1.33186650375182147021541603
2553621374194589058196151495680800614293386096089109 1.30649710484484829600
2076042240343358547735861789420714220240275665125516366937]
 [-2.4041501568734569628348958966755701250168148945022348708743612740191943
96993584 -3.321450152020386140950470842075654165712522189894878518271857911
438599824446073; -0.8854614151896057024966671879454477570212836844905008908
947440406019573571529885 -1.32852251348253439350326195978730271766772904065
6867292391218740689569332778719; 1.8922011306553324073623855292636045129954
07426909096337659203605619448511026197 3.4046264751891353010000851289504516
15830767176294298028177875189975702743336636; 2.168344641682897117532834671
401945741324777043825239001771240621167728487352111 0.910586991797430570647
2657969430378373023188575415925006219982874827525229517734]
 [-6.3175018941577900432839363826427714735358506541631321159777486369194914
3139909 -4.3155678039570944110518180008491453767976660370895940318348429119
98698391844232; -1.39642968227926987900863499511560676501460161941479899417
305503821385392618792 1.306665065142873131577948949498476462285785429265636
420061318141253190905763017; 7.90949557370585226967660686019406652155353584
9400571368505006877694816680882433 8.43856524736932942294592708225272169199
777781892396044333644507076528787228946; 0.52807480928563185986669807046145
29311181187439459818475083131900699987338018308 -1.686727457404011718833167
34375013560445033742492533244018157827339989626041513]
 [-6.4713586148365916404171985757833185421279199191702296115685350131723223
57615734 -1.075819223131073901514454191140344463200784526048488683193875189
027236558229648; 3.33545957795870527248367704334214500187659201877265503676
2223630204125059342682 8.05653499295683750289745578289805024295549243014088
6266600274113817392912991533; 14.070225572817499051204813994777992362212673
26656820249377270720884073654097473 10.597655405150589246887477524906463040
35424530822453804229358416455031634551036; -3.36483231761252065649807022164
5349993758302279715460465379703650299568605350405 -5.1347179891691693686785
43322510903527272528761710344723660785747960179801338579]
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
t: 10-element Vector{Float64}:
 0.0
 0.04618637081480677
 0.12086365528807747
 0.2104532924355688
 0.32245109880755135
 0.4456117029562153
 0.5897785541390024
 0.7396263389545401
 0.9155495882520033
 1.0
u: 10-element Vector{StaticArrays.SMatrix{4, 2, Float64, 8}}:
 [0.21270118893514223 0.0777629302598033; 0.9176800092815702 0.378198849378
40453; 0.37039279463209995 0.5341702264725026; 0.9646312146718936 0.1163800
6467451412]
 [-0.019918680140773498 0.04846146221738279; 0.7851430870030198 0.431235573
63559595; 0.3994313992046046 0.5287796663709102; 1.0861799855159138 0.16205
634075995212]
 [-0.4725070856428373 -0.022661377210716413; 0.4956498300961749 0.482488471
21724897; 0.5577380381216037 0.5386898993849728; 1.2308863401552983 0.22175
90013802414]
 [-1.1147249156827304 -0.1399003873553816; 0.0823386989229214 0.50020373342
55499; 0.9538249930165337 0.5889684015876882; 1.301400351519006 0.264286589
7386006]
 [-1.9988919895931987 -0.31575484870771986; -0.40108644600156457 0.48791118
641795495; 1.791783424956805 0.7204397399908864; 1.195909476302936 0.262087
0347781957]
 [-2.93081688633061 -0.5063876813914927; -0.6605781630395036 0.489583639803
1456; 3.1371002919625495 0.952090389476796; 0.7851475236676122 0.1762101621
0172596]
 [-3.677140939230487 -0.6463190725303357; -0.2313713535777358 0.61076400763
37647; 5.121744699875027 1.3021697678345143; -0.13604454130721377 -0.044956
93512597096]
 [-3.617765285180806 -0.5805801665875109; 1.4672411043782265 0.984124510648
0621; 7.239267838881068 1.6516191035811243; -1.6000001799785757 -0.40532768
08005405]
 [-1.765449811333309 -0.05737104619429323; 5.443744129391282 1.835000502239
0028; 8.795935797652216 1.7917881638579454; -3.8324446616024317 -0.94333211
33233372]
 [0.018551834709309434 0.41154421982140643; 8.09195615071357 2.393245586896
7764; 8.736522543847359 1.6452507753449608; -4.987682087234741 -1.210049039
9789736]
```



```julia
sol[3]
```

```
4×2 StaticArrays.SMatrix{4, 2, Float64, 8} with indices SOneTo(4)×SOneTo(2)
:
 -0.472507  -0.0226614
  0.49565    0.482488
  0.557738   0.53869
  1.23089    0.221759
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
  JULIA_DEPOT_PATH = /root/.cache/julia-buildkite-plugin/depots/a6029d3a-f78b-41ea-bc97-28aa57c6c6ea
  JULIA_NUM_THREADS = 16

```

Package Information:

```
      Status `/var/lib/buildkite-agent/builds/6-amdci4-julia-csail-mit-edu/julialang/scimltutorials-dot-jl/tutorials/introduction/Project.toml`
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
      Status `/var/lib/buildkite-agent/builds/6-amdci4-julia-csail-mit-edu/julialang/scimltutorials-dot-jl/tutorials/introduction/Manifest.toml`
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

