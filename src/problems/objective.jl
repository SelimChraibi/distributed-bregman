using Distributed
@everywhere using DistributedArrays
@everywhere using LinearAlgebra
@everywhere using Distributions

struct Objective
    n::Int64
    m::Int64
    A::DArray{Float64, 2}
    b::DArray{Float64, 2}
    L::Float64
    λ::Float64
    f::Function
    ∇f::Function
    g::Function
    ∇g::Function
    objective::Function
    ∇objective::Function
    function Objective(A::DArray{Float64,2},b::DArray{Float64,2},λ::Float64)

        L = norm(A,1) 
        n,m = size(A)
        
        @everyworker worker_objective = WorkerObjective($n÷nworkers(),$m,$A.localpart,$b.localpart)
        
        function f(x::Array{Float64, 2})
            1/nworkers() * @distributed (+) for worker = workers()
                worker_objective.f(x)
            end 
        end
        
        function ∇f(x::Array{Float64, 2})
            1/nworkers() * @distributed (+) for worker = workers()
                worker_objective.∇f(x)
            end
        end
        
        function g(x::Array{Float64, 2})
            λ * norm(x,1)
        end
        
        function ∇g(x::Array{Float64, 2})
            λ .* sign.(x)
        end
        
        function objective(x::Array{Float64, 2})
            f(x) + g(x) 
        end
        
        function ∇objective(x::Array{Float64, 2})
            ∇f(x) + ∇g(x) 
        end
        
        new(n,m,A,b,L,λ,f,∇f,g,∇g,objective,∇objective)
    end
end 

@everyworker struct WorkerObjective
    n::Int64
    m::Int64
    A::Matrix{Float64}
    b::Matrix{Float64}
    L::Float64
    f::Function
    ∇f::Function
    function WorkerObjective(n::Int64,m::Int64,A::Matrix{Float64},b::Matrix{Float64})
        L = norm(A,1) 
        function f(x::Array{Float64, 2})
            s = 0
            for i in 1:n
                aix = A[i,:]⋅x
                s = aix * (log(aix) - log(b[i]) - 1) + b[i] 
            end
            s
        end
        function ∇f(x::Array{Float64, 2})
            s = zeros(size(x))    
            for i in 1:n
                ai = A[i,:]
                s += ai * log(ai⋅x / b[i])
            end    
            s
        end
        new(n,m,A,b,L,f,∇f)
    end
end