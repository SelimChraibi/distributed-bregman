@everywhere using TickTock
ENV["TICKTOCK_MESSAGES"] = false

export optimize

function sync_optimize(x::Matrix{Float64}, solver::AbstractSolver; 
                  slow_workers=Dict{Int64, Int64}()::Dict{Int64, Int64}, 
                  epochs=0::Int64, time=0.0::Float64, verbose=0::Int64)
    network = Network() # initializes worker_network
    history = History(x, verbose) # initializes worker_history
    send(MasterPacket(x), network)
    
    tick()
    @async @everyworker while isopen(worker_network.down_remote_channel)
        local packet = receive(worker_network)
        local t = @elapsed local update = worker_solver.step(packet.x, worker_history)
        myid() in keys($slow_workers) && sleep($slow_workers[myid()]*t)
        log!(worker_history, packet.x, update)
        send(WorkerPacket(update), worker_network)
    end


    while (epochs==0 || history.epoch < epochs) && (time==0.0 || peektimer()<time)
        update = zeros(size(x))
        for worker in workers()
            packet = receive(network)
            update += packet.update
        end
        x = solver.step(update, history)
        log!(history, 0, x, peektimer())
        send(MasterPacket(x), network)
    end
    tok()
        
    @everyworker close(worker_network)
    close(network)
    history 
end

function async_optimize(x::Matrix{Float64}, solver::AbstractSolver; 
                  slow_workers=Dict{Int64, Int64}()::Dict{Int64, Int64}, 
                  epochs=0::Int64, time=0.0::Float64, verbose=0::Int64)
    
    network = Network() # initializes worker_network
    history = History(x, verbose) # initializes worker_history
    send(MasterPacket(x), network)
    
    tick()
    @async @everyworker while isopen(worker_network.down_remote_channel)
        local packet = receive(worker_network)
        local t = @elapsed local update = worker_solver.step(packet.x, worker_history)
        myid() in keys($slow_workers) && sleep($slow_workers[myid()]*t)
        log!(worker_history, packet.x, update)
        send(WorkerPacket(update), worker_network)
    end


    while (epochs==0 || history.epoch < epochs) && (time==0.0 || peektimer()<time)
        packet = receive(network)
        x = solver.step(packet.update, history)
        log!(history, packet.worker, x, peektimer())
        send(MasterPacket(x), packet.worker, network)
    end
    tok()
        
    @everyworker close(worker_network)
    close(network)
    history 
end
