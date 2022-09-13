using Distributed
@everywhere using DistributedArrays
@everywhere using LinearAlgebra
@everywhere using Distributions

struct PaperObjective
    n::Int64
    m::Int64
    A::DArray{Float64, 2}
    b::DArray{Float64, 2}
    L::Float64
    f::Function
    ∇f::Function
    x_opt::Matrix{Float64}
    λ::Float64
    function PaperObjective(A::DArray{Float64,2},b::DArray{Float64,2},λ::Float64)

        L = norm(A,1) 
        n,m = size(A)
        
        @everyworker worker_objective = WorkerPaperObjective($n÷nworkers(),$m,$A.localpart,$b.localpart)
        
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
        new(n,m,A,b,L,f,∇f,x_opt,λ)
    end
end 

@everyworker struct WorkerPaperObjective
    n::Int64
    m::Int64
    A::Matrix{Float64}
    b::Matrix{Float64}
    L::Float64
    f::Function
    ∇f::Function
    function WorkerPaperObjective(n::Int64,m::Int64,A::Matrix{Float64},b::Matrix{Float64})
        L = norm(A,1)
        function f(x::Array{Float64, 2})
            s = 0
            @inbounds @simd for i in 1:n
                ai = A[i,:]
                s += ai⋅x * log(ai⋅x) 
                s -= log(b[i]+1) * ai⋅x
                s += b[i] 
            end    
            s
        end
        function ∇f(x::Array{Float64, 2})
            s = zeros(size(x))    
            @inbounds @simd for i in 1:n
                ai = A[i,:]
                s += ai * log(ai⋅x / b[i])
            end    
            s
        end
        new(n,m,A,b,L,f,∇f)
    end
end