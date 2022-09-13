mutable struct History
    epoch::Int64
    iteration::Int64
    updates_per_worker::Dict{Int64, Int64}
    logs::Dict{String,Any}
    function History(x)
        epoch = 1
        iteration = 1
        updates_per_worker = Dict(worker=>0 for worker in workers())
        logs = Dict{String,Any}()
        logs["x"] = [copy(x)]
        logs["elapsed"] = [0.0]
        logs["epochs"] = [1]
        @everyworker (worker_history = WorkerHistory($x))
        new(epoch, iteration, updates_per_worker, logs)
    end
end

function log!(history::History, worker::Int64, x::Matrix{Float64}, elapsed::Float64, verbose=0::Int64)
    
    verbose!=0 && history.iteration==1 && println("epoch=$(history.epoch), elapsed=$(history.logs["elapsed"][end])")
    
    if worker == -1
        for worker in keys(history.updates_per_worker)
            history.updates_per_worker[worker] += 1
        end
    else
        history.updates_per_worker[worker] += 1
    end
    
    history.iteration += 1
    append!(history.logs["x"], [copy(x)])
    append!(history.logs["elapsed"], [elapsed])
    
    if all(values(history.updates_per_worker) .> 2*history.epoch)
        append!(history.logs["epochs"], [history.iteration])
        history.epoch += 1
        verbose!=0 && history.epoch%verbose==0 && println("epoch=$(history.epoch), elapsed=$(history.logs["elapsed"][end])")
    end
end

function log!(history::History, f::Function) 
    history.logs[string(f)] = []
    @inbounds @simd for x in history.logs["x"] # map ?
        append!(history.logs[string(f)],[f(x)]) 
    end
end

@everyworker mutable struct WorkerHistory
    logs::Dict{String,Vector{Matrix{Float64}}}
    function WorkerHistory(x)
        logs = Dict{String,Any}()
        logs["x"]=[copy(x)]
        logs["update"]=[copy(x)]
        new(logs)
    end
end

@everyworker function log!(worker_history::WorkerHistory, x::Matrix{Float64}, update::Matrix{Float64})
    append!(worker_history.logs["x"], [copy(x)]) 
    append!(worker_history.logs["update"], [copy(update)]) 
end

################################################

function last!(history::History, log_name::String; default=nothing::Any)
    if haskey(history.logs, log_name)
        return last(history.logs[log_name])
    else
        history.logs[log_name] = [copy(default)]
        return copy(default)
    end
end

function log!(history::History, log_name::String, log::Any)
    if haskey(history.logs, log_name) 
        append!(history.logs[log_name], [copy(log)])
    else
        history.logs[log_name] = [copy(log)]
    end
end

@everyworker function last!(worker_history::WorkerHistory, log_name::String; default=nothing::Any)
    if haskey(worker_history.logs, log_name)
        return last(worker_history.logs[log_name])
    else
        worker_history.logs[log_name] = [copy(default)]
        return copy(default)
    end
end

@everyworker function log!(worker_history::WorkerHistory, log_name::String, log::Any)
    if haskey(worker_history.logs, log_name) 
        append!(worker_history.logs[log_name], [copy(log)])
    else
        worker_history.logs[log_name] = [copy(log)]
    end
end