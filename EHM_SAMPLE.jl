# 05/23/2022 
# Apply rotation to Infinite MPS before generating samples
# Read Infinite MPS from previous VUMPS simulation

using Base: Float64
using LinearAlgebra
using ITensors 
using ITensorInfiniteMPS
using ITensors.HDF5
using Random

BLAS.set_num_threads(1)
ITensors.Strided.set_num_threads(1)
# ITensors.enable_threaded_blocksparse()
ITensors.disable_threaded_blocksparse()

# Define simulation parameters
model_params = (U=10.0, V=)

# Rotate an infinite MPS about Sy axis
function ITensors.op(::OpName"Ry", ::SiteType"Electron"; θ)
  return [
    1 0 0 0
    0 cos(θ) sin(θ) 0
    0 -sin(θ) cos(θ) 0
    0 0 0 1
  ]
end

# Read an infinite MPS from previous VUMPS results
# file = h5open("/mnt/home/bxiao/ceph/machine_learning/extended_hubbard/sampling/N32_Update/U8_BOND2000/Data/EH_U$(model_params[1])V$(model_params[2]).h5", "r")
file = h5open("../EH_U$(model_params[1])V$(model_params[2]).h5", "r")
ψ = read(file, "Infinite MPS", InfiniteCanonicalMPS)
# @show eltype(ψ)
# @show eltype(ψ.AL[1])
close(file)
output_file = h5open("../Sample_U$(model_params[1])V$(model_params[2]).h5", "w")

# Generate samples from an infinite MPS 
function sample(ψ::InfiniteCanonicalMPS, chain_length::Int)
  result = zeros(Int, chain_length)

  # Formulate finite MPS from infinite MPS.
  ψ1 = ψ.C[0] * ψ.AR[1]
  ϕ = MPS([ψ1, ψ.AR[2: chain_length]...])
  orthogonalize!(ϕ, 1)
  # @show ϕ
  
  if orthoCenter(ϕ) != 1   
      error("sample: Infinite MPS must have orthocenter == 1")
  end
  if abs(1.0 - norm(ϕ[1])) > 1E-8
      error("sample: MPS is not normalized, norm=$(norm(ψ[1]))")
  end

  A = ϕ[1]
  # @show A
  sites = [siteind(ψ, n) for n in 1:chain_length]
  for j in 1:chain_length
      s = sites[j]
      d = dim(s)
      pdisc = 0.0
      r = rand()
      # println("The random number is $(r)")

      n = 1
      An = ITensor()
      pn = 0.0

      while n <= d
          projn = ITensor(s)
          projn[s => n] = 1.0
          An = A * dag(projn) # noprime(A * U(theta)) * dag(projn) 
          pn = real(scalar(dag(An) * An))
          # println("The probablity computed is $(pn)")
          pdisc += pn
          (r < pdisc) && break
          n += 1
      end
      result[j] = n

      if j < chain_length
          A = ϕ[j + 1] * An
          A *= (1.0 / sqrt(pn))
      end 
  end
  return result
end

# Remove "Sz" conservation
rqn = "Sz"
for n in 1:nsites(ψ)
  ψ.AR[n] = removeqn(ψ.AR[n], rqn)
  ψ.AL[n] = removeqn(ψ.AL[n], rqn)
  ψ.C[n] = removeqn(ψ.C[n], rqn)
end


# Compute local observables e.g. Sz and densities
function ITensors.expect(ψ::InfiniteCanonicalMPS, o, n)
  return (noprime(ψ.AL[n] * ψ.C[n] * op(o, siteind(ψ, n))) * dag(ψ.AL[n] * ψ.C[n]))[]
end

# Generate samples from rotated infinite MPS
sample_number = 50000
site_number = 32

# Generate arrays to store 
sample_container = zeros(Int64, (sample_number, site_number))
Nup_container = zeros(Float64, (sample_number, 4))
Ndn_container = zeros(Float64, (sample_number, 4))
Ntotal_container = zeros(Float64, (sample_number, 4))
Sz_container = zeros(Float64, (sample_number, 4)) 


Threads.@threads for ind in 1:sample_number
  # Make a copy of the orginal \psi and rotate it
  ψ_copy = copy(ψ)
  θ = 2 * π * rand(Float64)
  @show θ
  flush(stdout)
  
  Ry = [op("Ry", siteind(ψ_copy, n); θ) for n in 1:nsites(ψ_copy)]
  for n in 1:nsites(ψ_copy)
    ψ_copy.AR[n] = apply(Ry[n], ψ_copy.AR[n])
    ψ_copy.AL[n] = apply(Ry[n], ψ_copy.AL[n])
  end

  Nup = [expect(ψ, "Nup", n) for n in 1:nsites(ψ)]
  Ndn = [expect(ψ, "Ndn", n) for n in 1:nsites(ψ)]
  Sz  = [expect(ψ, "Sz", n) for n in 1:nsites(ψ)]

  Nup_copy = [expect(ψ_copy, "Nup", n) for n in 1:nsites(ψ_copy)]
  Ndn_copy = [expect(ψ_copy, "Ndn", n) for n in 1:nsites(ψ_copy)]
  Sz_copy = [expect(ψ_copy, "Sz", n) for n in 1:nsites(ψ_copy)]

  # @show Nup, Ndn, Sz
  # @show Nup_copy, Ndn_copy, Sz_copy
  # @show Nup .+ Ndn, Nup_copy .+ Ndn_copy

  sample_container[ind, :] = sample(ψ_copy, site_number)
  Nup_container[ind, 1:2] = Nup;           Nup_container[ind, 3:4] = Nup_copy
  Ndn_container[ind, 1:2] = Ndn;           Ndn_container[ind, 3:4] = Ndn_copy
  Ntotal_container[ind, 1:2] = Nup .+ Ndn; Ntotal_container[ind, 3:4] = Nup_copy .+ Ndn_copy
  Sz_container[ind, 1:2] = Sz;             Sz_container[ind, 3:4] = Sz_copy
end

# @show sample_container
@show Sz_container
@show Ntotal_container
@show sample_container

# Store output data in a hdf5 file 
write(output_file, "Samples", sample_container)
write(output_file, "Nup", Nup_container)
write(output_file, "Ndn", Ndn_container)
write(output_file, "Ntotal", Ntotal_container)
write(output_file, "Sz", Sz_container)
close(output_file)

nothing