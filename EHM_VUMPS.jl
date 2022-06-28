# 06/09/2022 
# Add unitary operations S(θ) = U(θ)⁺SᶻU(θ) before generating samples [DONE] 
# Add performance evaluation tool @time; At large bond dimension, use imaginary-time evolution using tdvp

using Core: stderr
using LinearAlgebra
using ITensors 
using ITensorInfiniteMPS
using ITensors.HDF5
using KrylovKit: eigsolve

BLAS.set_num_threads(1)
ITensors.Strided.set_num_threads(1)
ITensors.enable_threaded_blocksparse()
# ITensors.disable_threaded_blocksparse()

########################################################################################################
# VUMPS parameters
#
maxdim = 200              # Maximum bond dimension
cutoff = 1e-8             # Singular value cutoff when increasing the bond dimension
max_vumps_iters = 100     # Maximum number of iterations of the VUMPS algorithm at each bond dimension
outer_iters =  7          # Number of times to increase the bond dimension
model_params = (t=1.0, U=6.0, V=)
########################################################################################################
# CODE BELOW HERE DOES NOT NEED TO BE MODIFIED

N = 2                # Unit cell size
sites_number = 32    # Number of sites truncated from infinite MPS
correlation_length = 1e-8
entropy0 = 0; entropy1 = 0
time_step = -10.0
output_file = h5open("../Data/EH_U$(model_params[2])V$(model_params[3]).h5", "w") 

########################################################################################################
# Initialize infinite MPS and MPO with Sz conservation
function electron_space_shift(q̃nf, q̃sz)
  return [
    QN(("Nf", 0 - q̃nf, -1), ("Sz", 0 - q̃sz)) => 1,
    QN(("Nf", 1 - q̃nf, -1), ("Sz", 1 - q̃sz)) => 1,
    QN(("Nf", 1 - q̃nf, -1), ("Sz", -1 - q̃sz)) => 1,
    QN(("Nf", 2 - q̃nf, -1), ("Sz", 0 - q̃sz)) => 1,
  ]
end
 
electron_space = fill(electron_space_shift(1, 0), N)
s = infsiteinds("Electron", N; space=electron_space)
initstate(n) = isodd(n) ? "↑" : "↓"
ψ = InfMPS(s, initstate)

# Build the Hamiltonian
model = Model"hubbard"()
H = InfiniteSum{MPO}(model, s; model_params...)
########################################################################################################


########################################################################################################
# Initialize infinite MPS and MPO and remove Sz conservation
# electron_space = fill(electron_space_shift(1, 0), N)
# s = infsiteinds("Electron", N; space=electron_space)
# rqn = "Sz"
# for n in 1:N
#   s[n] = removeqn(s[n], rqn)
# end
# initstate(n) = isodd(n) ? "↑" : "↓"
# ψ = InfMPS(s, initstate)
# @show ψ
# H = InfiniteSum{MPO}(model, s; model_params...)
########################################################################################################

# Check translational invariance
println("\nCheck translational invariance of initial infinite MPS")
@show norm(contract(ψ.AL[1:N]..., ψ.C[N]) - contract(ψ.C[0], ψ.AR[1:N]...))

println("\nRun VUMPS on initial product state, unit cell size $N")
outputlevel = 1
vumps_kwargs = (tol=1e-8, maxiter=max_vumps_iters, eigsolve_tol = (x -> x / 1000), outputlevel=outputlevel)
ψ = vumps(H, ψ; vumps_kwargs...)
flush(stdout); flush(stderr)

# Define bond dimensions, cutoff values and number of VUMPS iterations
bonds = [200, 200, 200, 200, 800, 1500, 2500]
vumps_iters = [50, 100, 100, 200, 200, 250, 250]
thresholds = [1e-7, 1e-7, 1e-8, 1e-8, 1e-8, 1e-8, 1e-8]

@time for iteration_index in 1:outer_iters
  println("\nIncrease bond dimension #$iteration_index")
  global maxdim = bonds[iteration_index]
  global max_vumps_iters = vumps_iters[iteration_index]
  global subspace_expansion_kwargs = (cutoff=cutoff, maxdim=maxdim)
  global vumps_kwargs = (tol=thresholds[iteration_index], maxiter=max_vumps_iters, eigsolve_tol = (x -> x / 1000), outputlevel=outputlevel)

  @time global ψ = subspace_expansion(ψ, H; subspace_expansion_kwargs...)
  println("\nRun VUMPS with new bond dimension")
  if iteration_index < outer_iters
    @time global ψ = vumps(H, ψ; vumps_kwargs...)
  else
    global vumps_kwargs = (tol=thresholds[iteration_index], maxiter=max_vumps_iters, outputlevel=outputlevel)
    @time global ψ = tdvp(H, ψ; time_step=time_step, vumps_kwargs...)
  end
  flush(stdout); flush(stderr)
end

# Check translational invariance
println("\nCheck translational invariance of optimized infinite MPS")
@show norm(contract(ψ.AL[1:N]..., ψ.C[N]) - contract(ψ.C[0], ψ.AR[1:N]...))


function ITensors.expect(ψ::InfiniteCanonicalMPS, o, n)
  return (noprime(ψ.AL[n] * ψ.C[n] * op(o, s[n])) * dag(ψ.AL[n] * ψ.C[n]))[]
end


function expect_two_site(ψ::InfiniteCanonicalMPS, h::ITensor, n1n2)
  n1, n2 = n1n2
  ϕ = ψ.AL[n1] * ψ.AL[n2] * ψ.C[n2]
  return (noprime(ϕ * h) * dag(ϕ))[]
end


# Add this function to measure energy per bond
# Based on the update from 2-local Hamiltonian to non-local Hamiltonian 
function expect_two_site(ψ::InfiniteCanonicalMPS, h::MPO, n1n2)
  return expect_two_site(ψ, prod(h), n1n2)
end

########################################################################################################
# Compute physical observables
Nup = [expect(ψ, "Nup", n) for n in 1:N]
Ndn = [expect(ψ, "Ndn", n) for n in 1:N]
Sz  = [expect(ψ, "Sz", n) for n in 1:N]
bs = [(1, 2), (2, 3)]
energy_infinite = map(b -> expect_two_site(ψ, H[b], b), bs)

# Print out local observables to check simulation results
println("\nResults from VUMPS")
@show energy_infinite
@show Nup
@show Ndn
@show Nup .+ Ndn
@show Sz
flush(stdout)

# Save ground-state wavefunction etc. to an output file 
write(output_file, "Infinite MPS", ψ)
write(output_file, "Infinite MPS Energy", energy_infinite)
write(output_file, "Infinite MPS Nup", Nup)
write(output_file, "Infinite MPS Ndn", Ndn)
write(output_file, "Infinite MPS Density", Nup .+ Ndn)
write(output_file, "Infinite MPS Sz", Sz) 
########################################################################################################


########################################################################################################
# Compute entanglement entropy 
function compute_entropy(input_matrix)
  local entropy = 0
  for index in 1 : size(input_matrix, 1)
      entropy += -2 * input_matrix[index, index]^2 * log(input_matrix[index, index])
  end
  return entropy
end

i0, j0 = inds(ψ.C[0])
_, C0, _ = svd(ψ.C[0], i0)
C0 = matrix(C0)
# @show ndims(C0); @show C0
entropy0 = compute_entropy(C0)
println("\n entropy is ", entropy0)
write(output_file, "Infinite MPS Entropy0", entropy0)

i1, j1 = inds(ψ.C[1])
_, C1, _ = svd(ψ.C[1], i1)
C1 = matrix(C1)
entropy1 = compute_entropy(C1)
println("\n entropy is ", entropy1)
write(output_file, "Infinite MPS Entropy1", entropy1)
########################################################################################################


########################################################################################################
# Compute correlation length using eigenvalues of transfer matrix
# println("\n")
T = TransferMatrix(ψ.AL)
Tᵀ = transpose(T)
# @show Tᵀ
vⁱᴿ = randomITensor(dag(input_inds(T)))
vⁱᴸ = randomITensor(dag(input_inds(Tᵀ)))

neigs = 10
tol = 1e-10
λ⃗ᴿ, v⃗ᴿ, right_info = eigsolve(T, vⁱᴿ, neigs, :LM; tol = tol)
λ⃗ᴸ, v⃗ᴸ, left_info = eigsolve(Tᵀ, vⁱᴸ, neigs, :LM; tol = tol)
@show norm(T(v⃗ᴿ[1]) - λ⃗ᴿ[1] * v⃗ᴿ[1]) 
@show norm(Tᵀ(v⃗ᴸ[1]) - λ⃗ᴸ[1] * v⃗ᴸ[1])

correlation_length = -1. / log((real(λ⃗ᴿ[2]) / real(λ⃗ᴿ[1]))) 
@show correlation_length
@show typeof(λ⃗ᴿ)
@show λ⃗ᴿ
@show λ⃗ᴸ
@show flux.(v⃗ᴿ)

write(output_file, "Infinite MPS Transfer Matrix Eigenvalue", λ⃗ᴿ)
write(output_file, "Infinite MPS Correlation Length", correlation_length)
########################################################################################################

nothing