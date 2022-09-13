using Distributed

export @everyworker

macro everyworker(ex)
    procs = GlobalRef(@__MODULE__, :procs)
    return esc(:($(Distributed).@everywhere $workers() $ex))
end