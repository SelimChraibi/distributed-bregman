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
    f::Function
    ∇f::Function
    λ::Float64
    function Objective(A::DArray{Float64,2},b::DArray{Float64,2},λ::Float64)

        L = norm(A,1) 
        n,m = size(A)
        
        @everyworker worker_objective = WorkerObjective($n÷nworkers(),$m,$A.localpart,$b.localpart)
        
        function f(x::Array{Float64, 2})
            @distributed (+) for worker = workers()
                worker_objective.f(x)
            end
        end

        function ∇f(x::Array{Float64, 2})
            @distributed (+) for worker = workers()
                worker_objective.∇f(x)
            end
        end
        new(n,m,A,b,L,f,∇f,λ)
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
                ai = A[i,:]
                s += ai⋅x * log(ai⋅x) 
                s -= log(b[i]+1) * ai⋅x
                s += b[i] 
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