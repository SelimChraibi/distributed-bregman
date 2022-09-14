using Distributed
using DistributedArrays


n_workers = 3
addprocs(n_workers,exeflags="--project");

include("./src/DistributedOptimization.jl")


n,m = 30, 20
A = [@spawnat worker rand(n÷nworkers(),m) for worker in workers()]
A = reshape(A, :, 1)
A = DArray(A); # +

x_opt = rand(m,1)

b = [@spawnat worker A.localpart*x_opt + 0.01*rand(Poisson(1),(n÷nworkers(),1)) for worker in workers()]
b = reshape(b, :, 1)
b = DArray(b); # ++

λ = 0.001

x = rand(m,1); epochs = 1000; verbose = epochs÷10;

paper_objective     = Objective(A,b,λ); # initializes worker_objective
paper_solver        = PaperSolver(paper_objective); # initializes worker_solver

history_async_paper = optimize(x, epochs, paper_solver, verbose);

iterations = history_async_paper.iteration; verbose = iterations÷10;
history_sync_paper  = sync_optimize(x, iterations, paper_solver, verbose);

x_star_paper = history_sync_paper.logs["x"][end];

piag_objective     = Objective(A,b,λ); # initializes worker_objective
piag_solver        = PiagSolver(piag_objective); # initializes worker_solver

history_async_piag = optimize(x, epochs, piag_solver, verbose);

iterations=history_async_piag.iteration; verbose = iterations÷10;
history_sync_piag  = sync_optimize(x, iterations, piag_solver, verbose);

x_star_piag = history_sync_piag.logs["x"][end];

# log!(history_async_paper, paper_objective.∇f)
# log!(history_sync_paper,  paper_objective.∇f)
# log!(history_async_piag, piag_objective.∇f)
# log!(history_sync_piag,  piag_objective.∇f)

using LinearAlgebra

h(x) = sum(x.*log.(x))
∇h(x) = 1 .+ log.(x)
D(h,∇h,x,y) = h(x) - h(y) - ∇h(y)⋅(x-y)

using Plots 
using PGFPlotsX
using LaTeXStrings


pgfplotsx()

# p = "Time"
p = "Epochs" 
# p = "Iterations"

plt = plot()
for (x_star, history, name) in zip([x_star_paper, x_star_piag],[history_sync_paper, history_sync_piag], ["sync_paper", "sync_piag"])
    if p=="Time"
        X = history.logs["elapsed"]
        Y = [D(h,∇h,x_star,x) for x in history.logs["x"]]
        plot!(X,Y,label=name)
    elseif p=="Epochs"
        Y = [D(h,∇h,x_star,x) for x in history.logs["x"]][history.logs["epochs"]]
        plot!(Y,label=name) 
    elseif p=="Iterations"
        Y = [D(h,∇h,x_star,x) for x in history.logs["x"]]
        plot!(Y,label=name) 
    end
end
    
xlabel!(p)
ylabel!(L"D_h(x_*,x_k)")

savefig(plt,"1.pdf")




# p = "Time"
# p = "Epochs" 
p = "Iterations"

plt = plot()
for (x_star, history, name) in zip([x_star_paper, x_star_piag],[history_async_paper, history_async_piag], ["async_paper", "async_piag"])
    if p=="Time"
        X = history.logs["elapsed"]
        Y = [D(h,∇h,x_star,x) for x in history.logs["x"]]
        plot!(X,Y,label=name)
    elseif p=="Epochs"
        Y = [D(h,∇h,x_star,x) for x in history.logs["x"]][history.logs["epochs"]]
        plot!(Y,label=name) 
    elseif p=="Iterations"
        Y = [D(h,∇h,x_star,x) for x in history.logs["x"]]
        plot!(Y,label=name) 
    end
end
    
xlabel!(p)
ylabel!(L"D_h(x_*,x_k)")

savefig(plt,"2.pdf")

rmprocs(workers())

# display(plt)
