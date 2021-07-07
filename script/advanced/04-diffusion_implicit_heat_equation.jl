
import Plots
using LinearAlgebra
using DiffEqBase
using OrdinaryDiffEq: SplitODEProblem, solve, IMEXEuler
import SciMLBase


a,b, n = 0, 1, 10               # zmin, zmax, number of cells
n̂_min, n̂_max = -1, 1            # Outward facing unit vectors
α = 100;                        # thermal diffusivity, larger means more stiff
β, γ = 10000, π;                # source term coefficients
Δt = 1000;                      # timestep size
N_t = 10;                       # number of timesteps to take
FT = Float64;                   # float type
Δz = FT(b-a)/FT(n)
Δz² = Δz^2;
∇²_op = [1/Δz², -2/Δz², 1/Δz²]; # interior Laplacian operator
∇T_bottom = 10;                 # Temperature gradient at the top
T_top = 1;                      # Temperature at the bottom
S(z) = β*sin(γ*z)               # source term, (sin for easy integration)
zf = range(a, b, length=n+1);   # coordinates on cell faces


function T_analytic(z) # Analytic steady state solution
    c1 = ∇T_bottom-β*cos(γ*a)/(γ*α)
    c2 = T_top-(β*sin(γ*b)/(γ^2*α)+c1*b)
    return β*sin(γ*z)/(γ^2*α)+c1*z+c2
end


# Initialize interior and boundary stencils:
∇² = Tridiagonal(
    ones(FT, n) .* ∇²_op[1],
    ones(FT, n+1)   .* ∇²_op[2],
    ones(FT, n) .* ∇²_op[3]
);

# Modify boundary stencil to account for BCs

∇².d[1] = -2/Δz²
∇².du[1] = +2/Δz²

# Modify boundary stencil to account for BCs
∇².du[n] = 0  # modified stencil
∇².d[n+1] = 0 # to ensure `∂_t T = 0` at `z=zmax`
∇².dl[n] = 0  # to ensure `∂_t T = 0` at `z=zmax`
D = α .* ∇²


AT_b = zeros(FT, n+1);
AT_b[1] = α*2/Δz*∇T_bottom*n̂_min;
AT_b[end-1] = α*T_top/Δz²;


T = zeros(FT, n+1);
T .= 1;
T[n+1] = T_top; # set top BC


function rhs!(dT, T, params, t)
    n = params.n
    i = 1:n # interior domain
    dT[i] .= S.(zf[i]) .+ AT_b[i]
    return dT
end;


params = (;n)

tspan = (FT(0), N_t*FT(Δt))

prob = SplitODEProblem(
    SciMLBase.DiffEqArrayOperator(
        D,
    ),
    rhs!,
    T,
    tspan,
    params
)
alg = IMEXEuler(linsolve=LinSolveFactorize(lu!))
println("Solving...")
sol = solve(
    prob,
    alg,
    dt = Δt,
    saveat = range(FT(0), N_t*FT(Δt), length=5),
    progress = true,
    progress_message = (dt, u, p, t) -> t,
);


T_end = sol.u[end]

p1 = Plots.plot(zf, T_analytic.(zf), label="analytic", markershape=:circle, markersize=6)
p1 = Plots.plot!(p1, zf, T_end, label="numerical", markershape=:diamond)
p1 = Plots.plot!(p1, title="T ∈ cell faces")

p2 = Plots.plot(zf, abs.(T_end .- T_analytic.(zf)), label="error", markershape=:circle, markersize=6)
p2 = Plots.plot!(p2, title="T ∈ cell faces")

Plots.plot(p1, p2)

