struct PaperSolver <: AbstractSolver
    step::Function
    objective::Objective
    γ::Float64
    function PaperSolver(objective::Objective, γ=1/objective.L::Float64)
        function step(update::Matrix{Float64}, history::History)
            ū = last!(history, "ū"; default=history.logs["x"][end])
            ū = ū + update/nworkers()
            m = objective.m
            x = zeros((m,1))
            for j in 1:m
                x[j] = 1/exp(1 + γ*objective.λ + ū[j])
            end
            log!(history, "ū", ū)
            return x
        end
        @everyworker worker_solver = WorkerPaperSolver(worker_objective, $γ)
        new(step, objective, γ)
    end
end

@everyworker struct WorkerPaperSolver
    step::Function
    worker_objective::WorkerObjective
    γ::Float64
    function WorkerPaperSolver(worker_objective::WorkerObjective, γ::Float64)
        function step(x::Matrix{Float64}, worker_history::WorkerHistory)
            u = last!(worker_history, "u"; default=worker_history.logs["x"][end])
            n,m = worker_objective.n, worker_objective.m
            A,b = worker_objective.A, worker_objective.b
            u₊ = zeros((m,1))
            
            for i in 1:n
                ai = A[i,:]
                bi = b[i]
                u₊ += ai * log(ai⋅x / bi)
            end
            
            u₊ =  γ * u₊ - (1 .+ log.(x))
            Δ = u₊ - u
            u = u₊
            log!(worker_history, "u", u)
            return Δ
        end
        new(step, worker_objective, γ)
    end
end