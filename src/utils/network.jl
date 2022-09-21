using Distributed
@everywhere using Distributed

export Network, close #, distributed_network


########################################

@everywhere mutable struct MasterPacket
    x::Matrix{Float64}
end

@everywhere mutable struct WorkerPacket
    update::Matrix{Float64}
    worker::Int64
    
    function WorkerPacket(update)
        new(update, myid())
    end
end

########################################

struct Network
    up_remote_channel::RemoteChannel
    down_remote_channels::Dict{Int64, RemoteChannel}

    function Network()
        up_remote_channel = RemoteChannel(()->Channel{WorkerPacket}(nworkers()), myid())
        down_remote_channels = Dict(worker => RemoteChannel(()->Channel{MasterPacket}(nworkers())) for worker in workers())
        
        @everyworker (worker_network = WorkerNetwork($up_remote_channel, $down_remote_channels[myid()]))
        new(up_remote_channel, down_remote_channels)
    end
end

@everyworker struct WorkerNetwork
    up_remote_channel::RemoteChannel
    down_remote_channel::RemoteChannel
end

########################################

# function distributed_network(f::Function)#, nworkers::Int64)
#     # nprocs()==1 || rmprocs(workers()); addprocs(nworkers)
#     # include(string(@__DIR__)*"/network_architecture.jl")
#     network = Network()
#     try
#         f(network)
#     finally
#         close(network)
#     end
# end

function Base.close(network::Network)
    close(network.up_remote_channel)
    close.(values(network.down_remote_channels))
end

@everyworker function Base.close(worker_network::WorkerNetwork)
    close(worker_network.up_remote_channel)
    close(worker_network.down_remote_channel)
    GC.gc()
end

########################################

function send(packet::MasterPacket, worker::Int64, network::Network)
    put!(network.down_remote_channels[worker], packet)
end

function send(packet::Any, network::Network)
    for worker in workers()
        put!(network.down_remote_channels[worker], packet)
    end
end

@everyworker function send(packet::WorkerPacket, worker_network::WorkerNetwork)
    put!(worker_network.up_remote_channel, packet)
end

function receive(network::Network)
    take!(network.up_remote_channel)
end

@everyworker function receive(worker_network::WorkerNetwork)
    take!(worker_network.down_remote_channel)
end


