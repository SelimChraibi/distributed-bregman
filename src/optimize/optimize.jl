@everywhere using TickTock
@everywhere ENV["TICKTOCK_MESSAGES"] = false

export optimize

function sync_optimize(x::Matrix{Float64}, iterations::Int64, solver::AbstractSolver, verbose=0::Int64)
    network = Network()
    history = History(x) #initializes worker_history
    
    tick()
    for iteration in 1:iterations
        update = @distributed (+) for worker in workers()
            update = worker_solver.step(x, worker_history)
            log!(worker_history, x, update)
            return update
        end 
        
        x = solver.step(update, history)
        log!(history, -1, x, peektimer(), verbose)
    end
    tok()
    
    close(network)
    history
end

function optimize(x::Matrix{Float64}, epochs::Int64, solver::AbstractSolver, verbose=0::Int64)
    network = Network()
    history = History(x) #initializes worker_history
    send(MasterPacket(x), network)
    slow_workers = nworkers() > 1 ? workers()[1:3] : [-1,-1,-1]

    @async @everyworker while isopen(worker_network.down_remote_channel)
        packet = receive(worker_network)
        t = @elapsed update = worker_solver.step(packet.x, worker_history)
        myid() == $slow_workers[1] && sleep(t*5)
        myid() == $slow_workers[2] && sleep(t*10)
        myid() == $slow_workers[3] && sleep(t*15)
        log!(worker_history, packet.x, update)
        send(WorkerPacket(update), worker_network)
    end
    
    tick()
    while history.epoch < epochs
        packet = receive(network)
        x = solver.step(packet.update, history)
        log!(history, packet.worker, x, peektimer(), verbose)
        send(MasterPacket(x), packet.worker, network)
    end
    tok()
    
    close(network)
    history
end