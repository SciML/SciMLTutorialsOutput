
using DiffEqJump
dims = (5,5)
num_nodes = prod(dims) # number of sites
grid = CartesianGrid(dims) # or use LightGraphs.grid(dims)


num_species = 3
starting_state = zeros(Int, num_species, num_nodes)
starting_state[1,1] = 25
starting_state[2,end] = 25
starting_state


tspan = (0.0, 3.0)
rates = [6.0, 0.05] # k_1 = rates[1], k_2 = rates[2]


prob = DiscreteProblem(starting_state, tspan, rates)


netstoch = [[1 => -1, 2 => -1, 3 => 1],[1 => 1, 2 => 1, 3 => -1]]
reactstoch = [[1 => 1, 2 => 1],[3 => 1]]
majumps = MassActionJump(rates, reactstoch, netstoch)


hopping_constants = ones(num_species, num_nodes)
hopping_constants[3, :] .= 0.0
hopping_constants


alg = NSM()
jump_prob = JumpProblem(prob, alg, majumps, hopping_constants=hopping_constants, spatial_system = grid, save_positions=(true, false))


solution = solve(jump_prob, SSAStepper())


using Plots
is_static(spec) = (spec == 3) # true if spec does not hop
"get frame k"
function get_frame(k, sol, linear_size, labels, title)
    num_species = length(labels)
    h = 1/linear_size
    t = sol.t[k]
    state = sol.u[k]
    xlim=(0,1+3h/2); ylim=(0,1+3h/2);
    plt = plot(xlim=xlim, ylim=ylim, title = "$title, $(round(t, sigdigits=3)) seconds")

    species_seriess_x = [[] for i in 1:num_species]
    species_seriess_y = [[] for i in 1:num_species]
    CI = CartesianIndices((linear_size, linear_size))
    for ci in CartesianIndices(state)
        species, site = Tuple(ci)
        x,y = Tuple(CI[site])
        num_molecules = state[ci]
        sizehint!(species_seriess_x[species], num_molecules)
        sizehint!(species_seriess_y[species], num_molecules)
        if !is_static(species)
            randsx = rand(num_molecules)
            randsy = rand(num_molecules)
        else
            randsx = zeros(num_molecules)
            randsy = zeros(num_molecules)
        end
        for k in 1:num_molecules
            push!(species_seriess_x[species], x*h - h/4 + 0.5h*randsx[k])
            push!(species_seriess_y[species], y*h - h/4 + 0.5h*randsy[k])
        end
    end
    for species in 1:num_species
        scatter!(plt, species_seriess_x[species], species_seriess_y[species], label = labels[species], marker = 6)
    end
    xticks!(plt, range(xlim...,length = linear_size+1))
    yticks!(plt, range(ylim...,length = linear_size+1))
    xgrid!(plt, 1, 0.7)
    ygrid!(plt, 1, 0.7)
    return plt
end

"make an animation of solution sol in 2 dimensions"
function animate_2d(sol, linear_size; species_labels, title, verbose = true)
    num_frames = length(sol.t)
    anim = @animate for k=1:num_frames
        verbose && println("Making frame $k")
        get_frame(k, sol, linear_size, species_labels, title)
    end
    anim
end
# animate
anim=animate_2d(solution, 5, species_labels = ["A", "B", "C"], title = "A + B <--> C", verbose = false)
fps = 5
gif(anim, fps = fps)


dims = (2,3,4) # can pass in a 1-Tuple, a 2-Tuple or a 3-Tuple
num_nodes = prod(dims)
grid = CartesianGrid(dims)


using LightGraphs
graph = cycle_digraph(5) # directed cyclic graph on 5 nodes


hopping_constants = Matrix{Vector{Float64}}(undef, num_species, num_nodes)
for ci in CartesianIndices(hopping_constants)
    (species, site) = Tuple(ci)
    hopping_constants[species, site] = zeros(outdegree(grid, site))
    for (n, nb) in enumerate(neighbors(grid, site))
        if nb < site
            hopping_constants[species, site][n] = 1.0
        end
    end
end


species_hop_constants = ones(num_species)
site_hop_constants = Vector{Vector{Float64}}(undef, num_nodes)
for site in 1:num_nodes
    site_hop_constants[site] = ones(outdegree(grid, site))
end
hopping_constants=Pair(species_hop_constants, site_hop_constants)


species_hop_constants = ones(num_species, num_nodes)
site_hop_constants = Vector{Vector{Float64}}(undef, num_nodes)
for site in 1:num_nodes
    site_hop_constants[site] = ones(outdegree(grid, site))
end
hopping_constants=Pair(species_hop_constants, site_hop_constants)

