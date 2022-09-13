struct PiagSolver <: AbstractSolver
    step::Function
    objective::PiagObjective
    function PiagSolver(objective::PiagObjective)
        function step(update::Matrix{Float64}, history::History, γ=1/objective.L::Float64)
            ū = last!(history, "ū"; default=history.logs["x"][end])
            ū = ū + update/nworkers()
            x₋ = copy(last(history.logs["x"]))
            m = objective.m
            x = zeros((m,1))
            @inbounds @simd for j in 1:m
                x[j] = 1/exp(γ*objective.λ + ū[j] - log(x₋[j]))
            end
            log!(history, "ū", ū)
            return x
        end
        @everyworker worker_solver = WorkerPiagSolver(worker_objective)
        new(step, objective)
    end
end

@everyworker struct WorkerPiagSolver
    step::Function
    worker_objective::WorkerPiagObjective
    function WorkerPiagSolver(worker_objective::WorkerPiagObjective, γ=1/worker_objective.L::Float64)
        function step(x::Matrix{Float64}, worker_history::WorkerHistory)
            u = last!(worker_history, "u"; default=worker_history.logs["x"][end])
            n,m = worker_objective.n, worker_objective.m
            A,b = worker_objective.A, worker_objective.b
            u₊ = zeros((m,1))
            
            @inbounds @simd for i in 1:n
                ai = A[i,:]
                bi = b[i]
                u₊ += ai * log(ai⋅x / bi)
            end
            
            u₊ *= γ
            Δ = u₊ - u
            u = u₊
            log!(worker_history, "u", u)
            return Δ
        end
        new(step, worker_objective)
    end
end