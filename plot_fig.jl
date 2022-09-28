################################
####### Required packages ######
################################
using JLD2, FileIO
using Plots
using PlotlyJS
using PGFPlotsX
using LaTeXStrings

################################
#### Activating the project ####
################################
include("./src/DistributedOptimization.jl")


###############################################################
########################## Loading ############################
###############################################################


file_name = "F"
L  = load("./data/$file_name.jld2")
histories = L["histories"]


###############################################################
########################### Plots #############################
###############################################################

objective = 0

function learning_curves(;xlabel, ylabel, yscale, histories, to_plot, objective=objective, file_name="", save_as="", x_star=nothing, ribbon=false)

    #save_as in ["tex", "pdf", "tikz"] ? pgfplotsx() : gr()
    pgfplotsx()
    plt = Plots.plot(yscale=yscale, legend=:topright, grid=true)
    
    h(x)  = x⋅log.(x)
    ∇h(x) = log.(x) .+ 1
    
    ylabel=="norm"        && (D = (x,y) -> norm(x-y,2); ylabel!(L"||x^*-x^k||_2"))
    ylabel=="bregman"     && (D = (x,y) -> h(x) - h(y) - ∇h(y)⋅(x-y); ylabel!(L"D_h(x^*,x^k)"))
    ylabel=="objective"   && ylabel!(L"f(x^k)+g(x^k)")
    ylabel=="∇objective"  && ylabel!(L"||∇f(x^k)+∇g(x^k)||^2")
    
    xlabel=="time"        && xlabel!("Time (s)")
    xlabel=="epochs"      && xlabel!("Epochs")
    xlabel=="iterations"  && xlabel!("Iterations")
    
    isnothing(x_star) && (x_stars = [history.logs["x"][end] for history in histories["sync_best"]])
    
    for algorithm in to_plot
        
        algo_histories = histories[algorithm]
        
        algorithm == "paper" && (linestyle=:solid; label="Ours"; color=:palegreen2)
        algorithm == "piag" && (linestyle=:solid; label="PIAG"; color=:salmon1)
        algorithm == "sync" && (linestyle=:solid; label="Synchronous"; color=:deepskyblue1)
        algorithm == "paper_best" && (linestyle=:dash; label="Ours (best γ)"; color=:palegreen2)
        algorithm == "piag_best" && (linestyle=:dash; label="PIAG (best γ)"; color=:salmon1)
        algorithm == "sync_best" && (linestyle=:dash; label="Synchronous (best γ)"; color=:deepskyblue1)
        
        X_tmp = []
        Y_tmp = []
        
        end_epoch       = min([history.epoch for history in algo_histories]...)
        end_time        = min([history.logs["elapsed"][end] for history in algo_histories]...)
        end_iteration   = min([length(history.logs["x_iter"]) for history in algo_histories]...)
        
        for (i, history) in enumerate(algo_histories)
            if xlabel=="iterations"
                isnothing(x_star) && (x_star = x_stars[i])
                append!(Y_tmp, [[D(x_star,x) for x in history.logs["x_iter"][1:end_iteration]]])
                
            else
                
                if ylabel=="objective"
                    !haskey(history.logs, "objective") && log!(history, "objective", objective.objective)
                    append!(Y_tmp, [history.logs["objective"][1:end_epoch]])
                elseif ylabel=="∇objective"
                    !haskey(history.logs, "∇objective") && log!(history, "∇objective", objective.∇objective)
                    append!(Y_tmp, [[norm(g,2) for g in history.logs["∇objective"]][1:end_epoch]])
                elseif ylabel in ["norm", "bregman"]
                    isnothing(x_star) && (x_star = x_stars[i])
                    append!(Y_tmp, [[D(x_star,x) for x in history.logs["x"][1:end_epoch]]])
                end
            
                xlabel=="time" && append!(X_tmp, [history.logs["elapsed"][1:end_epoch]])
            end
            
        end
        
        if xlabel=="epochs"
            X = collect(1:end_epoch)
            Y = mean(Y_tmp)
            V = std(Y_tmp)
        elseif xlabel=="iterations" 
            X = collect(1:end_iteration)
            Y = mean(Y_tmp)
            V = std(Y_tmp)
        elseif xlabel=="time"
            X = mean(X_tmp)
            Y = mean(Y_tmp)[X .< end_time]
            V = std(Y_tmp)[X .< end_time]
            X = X[X .< end_time]
        end

        plotted_points = floor.(Int,collect(LinRange(1,length(Y),200)))
        

        if !ribbon || (yscale  == :log && any(Y.-V .== 0))
            plot!(X[plotted_points], Y[plotted_points], label=label, linestyle=linestyle, linewidth=2, color=color)
        else
            plot!(X[plotted_points], Y[plotted_points], ribbon=V, fillalpha=.3, label=label, linewidth=2, linestyle=linestyle, color=color)  
        end

    end
    
    isdir("./plots") || mkdir("./plots");
    save_as!="" && Plots.savefig(plt, "./plots/$file_name.$save_as")
    
end


plt = learning_curves(xlabel    = "time", 
                      ylabel    = "bregman",
                      yscale    = :log,
                      histories = histories,
                      to_plot   = ["sync", "piag", "paper"],
                      file_name = "comp",
                      save_as   = "png",
                      #x_star    = x_star,
                      ribbon    = false)


plt = learning_curves(xlabel    = "time", 
                      ylabel    = "bregman",
                      yscale    = :log,
                      histories = histories,
                      to_plot   = ["piag_best", "paper_best","sync_best"],
                      file_name = "comp_tuned",
                      save_as   = "png",
                      #x_star    = x_star,
                      ribbon    = false)


plt = learning_curves(xlabel    = "time", 
                      ylabel    = "bregman",
                      yscale    = :log,
                      histories = histories,
                      to_plot   = ["sync", "piag", "paper"],
                      file_name = "comp",
                      save_as   = "tikz",
                      #x_star    = x_star,
                      ribbon    = false)


plt = learning_curves(xlabel    = "time", 
                      ylabel    = "bregman",
                      yscale    = :log,
                      histories = histories,
                      to_plot   = ["piag_best", "paper_best"],
                      file_name = "comp_tuned",
                      save_as   = "tikz",
                      #x_star    = x_star,
                      ribbon    = false)

pltA = learning_curves(xlabel   = "iterations", 
                      ylabel    = "bregman",
                      yscale    = :log,
                      histories = histories,
                      to_plot   = ["paper"],
                      file_name = "A",
                      save_as   = "png",
                      #x_star    = x_star,
                      ribbon    = false);

pltB = learning_curves(xlabel   = "iterations", 
                      ylabel    = "norm",
                      yscale    = :log,
                      histories = histories,
                      to_plot   = ["paper"],
                      file_name = "B",
                      save_as   = "png",
                      #x_star    = x_star,
                      ribbon    = false);

pltC = learning_curves(xlabel   = "time", 
                      ylabel    = "bregman",
                      yscale    = :log,
                      histories = histories,
                      to_plot   = ["paper", "sync"],
                      file_name = "C",
                      save_as   = "png",
                      #x_star    = x_star,
                      ribbon    = false);


pltA = learning_curves(xlabel   = "iterations", 
                      ylabel    = "bregman",
                      yscale    = :log,
                      histories = histories,
                      to_plot   = ["paper"],
                      file_name = "A",
                      save_as   = "tikz",
                      #x_star    = x_star,
                      ribbon    = false);

pltB = learning_curves(xlabel   = "iterations", 
                      ylabel    = "norm",
                      yscale    = :log,
                      histories = histories,
                      to_plot   = ["paper"],
                      file_name = "B",
                      save_as   = "tikz",
                      #x_star    = x_star,
                      ribbon    = false);

pltC = learning_curves(xlabel   = "time", 
                      ylabel    = "bregman",
                      yscale    = :log,
                      histories = histories,
                      to_plot   = ["paper", "sync"],
                      file_name = "C",
                      save_as   = "tikz",
                      #x_star    = x_star,
                      ribbon    = false);


# pgfplotsx()
# plt = Plots.plot(pltA, pltB, pltC, layout=grid(1,3))
# Plots.savefig(plt, "./plots/theorem_illustration.tikz")
# Plots.savefig(plt, "./plots/theorem_illustration.png")
# plt


