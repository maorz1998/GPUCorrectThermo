#include "dfThermo.h"
#include <filesystem>
#include <cmath>
#include <numeric>
#include <cassert>
#include <cstring>
#include "device_launch_parameters.h"

__global__ void get_mole_fraction_mean_mole_weight(int num_cells, int num_species, const double *d_Y, 
        const double *molecular_weight, double *mole_fraction, double *mean_mole_weight)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num_cells)
        return;
    
    double sum = 0.;
    for (int i = 0; i < num_species; i++) {
        sum += d_Y[i * num_cells + index] / molecular_weight[i];
    }
    double meanMoleWeight = 0.;
    for (int i = 0; i < num_species; i++) {
        mole_fraction[i * num_cells + index] = d_Y[i * num_cells + index] / (molecular_weight[i] * sum);
        meanMoleWeight += mole_fraction[i * num_cells + index] * molecular_weight[i];
    }
    mean_mole_weight[index] = meanMoleWeight;
}

__global__ void calculate_TPoly_kernel(int num_cells, const double *T, double *d_T_poly)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num_cells)
        return;
    
    d_T_poly[num_cells * 0 + index] = 1.0;
    d_T_poly[num_cells * 1 + index] = log(T[index]);
    d_T_poly[num_cells * 2 + index] = d_T_poly[num_cells * 1 + index] * d_T_poly[num_cells * 1 + index];
    d_T_poly[num_cells * 3 + index] = d_T_poly[num_cells * 1 + index] * d_T_poly[num_cells * 2 + index];
    d_T_poly[num_cells * 4 + index] = d_T_poly[num_cells * 2 + index] * d_T_poly[num_cells * 2 + index];
}

__global__ void calculate_psi_kernel(int num_cells, const double *T, const double *mean_mole_weight,
        double *psi)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num_cells)
        return;
    
    psi[index] = mean_mole_weight[index] / (GAS_CANSTANT * T[index]);
}

__global__ void calculate_rho_kernel(int num_cells, const double *p, const double *psi, double *rho)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num_cells)
        return;
    
    rho[index] = p[index] * psi[index];
}

__global__ void calculate_viscosity_kernel(int num_cells, int num_species, const double *viscosity_coeffs,
        const double *T_poly, const double *T, const double *mole_fraction, const double *molecular_weights, 
        double *species_viscosities, double *mu)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num_cells)
        return;
    
    double dot_product;
    double local_T = T[index];

    for (int i = 0; i < num_species; i++) {
        dot_product = 0.;
        for (int j = 0; j < 5; j++) {
            dot_product += viscosity_coeffs[i * 5 + j] * T_poly[num_cells * j + index];
        }
        species_viscosities[i * num_cells + index] = (dot_product * dot_product) * sqrt(local_T);
    }

    double mu_mix = 0.;
    for (int i = 0; i < num_species; i++) {
        double temp = 0.;
        for (int j = 0; j < num_species; j++) {
            temp += mole_fraction[num_cells * j + index] / pow(8, 0.5) *
            pow((1 + molecular_weights[i] / molecular_weights[j]), -0.5) *
            pow(1.0 + sqrt(species_viscosities[i * num_cells + index] / species_viscosities[j * num_cells + index]) *
            pow(molecular_weights[j] / molecular_weights[i], 0.25), 2.0);
        }
        mu_mix += mole_fraction[num_cells * i + index] * species_viscosities[i * num_cells + index] / temp;
    }

    mu[index] = mu_mix;
}

__device__ double calculate_cp_device_kernel(int num_cells, int num_species, int index, const double *nasa_coeffs, 
        const double local_T, const double *mass_fraction, const double *molecular_weights)
{   
    double cp = 0.;

    for (int i = 0; i < num_species; i++) {
        if (local_T > nasa_coeffs[i * 15 + 0]) {
            cp += mass_fraction[i * num_cells + index] * (nasa_coeffs[i * 15 + 1] + nasa_coeffs[i * 15 + 2] * local_T + nasa_coeffs[i * 15 + 3] * local_T * local_T + 
                    nasa_coeffs[i * 15 + 4] * local_T * local_T * local_T + 
                    nasa_coeffs[i * 15 + 5] * local_T * local_T * local_T * local_T) * GAS_CANSTANT / molecular_weights[i];
        } else {
            cp += mass_fraction[i * num_cells + index] * (nasa_coeffs[i * 15 + 8] + nasa_coeffs[i * 15 + 9] * local_T + nasa_coeffs[i * 15 + 10] * local_T * local_T + 
                    nasa_coeffs[i * 15 + 11] * local_T * local_T * local_T + 
                    nasa_coeffs[i * 15 + 12] * local_T * local_T * local_T * local_T) * GAS_CANSTANT / molecular_weights[i];
        }
    }
    return cp;
}

__global__ void calculate_thermoConductivity_kernel(int num_cells, int num_species, 
        const double *nasa_coeffs, const double *mass_fraction,
        const double *thermal_conductivity_coeffs, const double *T_poly, const double *T,
        const double *mole_fraction, const double *molecular_weights, 
        double *species_thermal_conductivities, double *thermal_conductivity)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num_cells)
        return;

    double dot_product;
    double local_T = T[index];

    for (int i = 0; i < num_species; i++) {
        dot_product = 0.;
        for (int j = 0; j < 5; j++) {
            dot_product += thermal_conductivity_coeffs[i * 5 + j] * T_poly[num_cells * j + index];
        }
        species_thermal_conductivities[i * num_cells + index] = dot_product * sqrt(local_T);
    }

    double sum_conductivity = 0.;
    double sum_inv_conductivity = 0.;

    for (int i = 0; i < num_species; i++) {
        sum_conductivity += mole_fraction[num_cells * i + index] * species_thermal_conductivities[i * num_cells + index];
        sum_inv_conductivity += mole_fraction[num_cells * i + index] / species_thermal_conductivities[i * num_cells + index];
    }
    double lambda_mix = 0.5 * (sum_conductivity + 1.0 / sum_inv_conductivity);

    double cp = calculate_cp_device_kernel(num_cells, num_species, index, nasa_coeffs, local_T, mass_fraction, molecular_weights);

    thermal_conductivity[index] = lambda_mix / cp;
}

__global__ void calculate_cp_global_kernel(int num_cells, int num_species, const double *nasa_coeffs, 
        const double *T, const double *mass_fraction, const double *molecular_weights, double *cp)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num_cells)
        return;
    
    double local_T = T[index];
    double local_cp = 0.;

    for (int i = 0; i < num_species; i++) {
        if (local_T > nasa_coeffs[i * 15 + 0]) {
            local_cp += mass_fraction[i * num_cells + index] * (nasa_coeffs[i * 15 + 1] + nasa_coeffs[i * 15 + 2] * local_T + nasa_coeffs[i * 15 + 3] * local_T * local_T + 
                    nasa_coeffs[i * 15 + 4] * local_T * local_T * local_T + 
                    nasa_coeffs[i * 15 + 5] * local_T * local_T * local_T * local_T) * GAS_CANSTANT / molecular_weights[i];
        } else {
            local_cp += mass_fraction[i * num_cells + index] * (nasa_coeffs[i * 15 + 8] + nasa_coeffs[i * 15 + 9] * local_T + nasa_coeffs[i * 15 + 10] * local_T * local_T + 
                    nasa_coeffs[i * 15 + 11] * local_T * local_T * local_T + 
                    nasa_coeffs[i * 15 + 12] * local_T * local_T * local_T * local_T) * GAS_CANSTANT / molecular_weights[i];
        }
    }
    cp[index] = local_cp;
}

__global__ void calculate_enthalpy_kernel(int num_cells, int num_species, const double *T,
        const double *nasa_coeffs, const double *molecular_weights, const double *mass_fraction, double *enthalpy)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num_cells)
        return;

    double h = 0.;
    double local_T = T[index];

    for (int i = 0; i < num_species; i++) {
        if (local_T > nasa_coeffs[i * 15 + 0]) {
            h += (nasa_coeffs[i * 15 + 1] + nasa_coeffs[i * 15 + 2] * local_T / 2 + nasa_coeffs[i * 15 + 3] * local_T * local_T / 3 + 
                    nasa_coeffs[i * 15 + 4] * local_T * local_T * local_T / 4 + nasa_coeffs[i * 15 + 5] * local_T * local_T * local_T * local_T / 5 + 
                    nasa_coeffs[i * 15 + 6] / local_T) * GAS_CANSTANT * local_T / molecular_weights[i] * mass_fraction[i * num_cells + index];
        } else {
            h += (nasa_coeffs[i * 15 + 8] + nasa_coeffs[i * 15 + 9] * local_T / 2 + nasa_coeffs[i * 15 + 10] * local_T * local_T / 3 + 
                    nasa_coeffs[i * 15 + 11] * local_T * local_T * local_T / 4 + nasa_coeffs[i * 15 + 12] * local_T * local_T * local_T * local_T / 5 + 
                    nasa_coeffs[i * 15 + 13] / local_T) * GAS_CANSTANT * local_T / molecular_weights[i] * mass_fraction[i * num_cells + index];
        }
    }
    enthalpy[index] = h;
}

__device__ double calculate_enthalpy_device_kernel(int num_cells, int num_species, int index, const double local_T,
        const double *nasa_coeffs, const double *molecular_weights, const double *mass_fraction)
{
    double h = 0.;

    for (int i = 0; i < num_species; i++) {
        if (local_T > nasa_coeffs[i * 15 + 0]) {
            h += (nasa_coeffs[i * 15 + 1] + nasa_coeffs[i * 15 + 2] * local_T / 2 + nasa_coeffs[i * 15 + 3] * local_T * local_T / 3 + 
                    nasa_coeffs[i * 15 + 4] * local_T * local_T * local_T / 4 + nasa_coeffs[i * 15 + 5] * local_T * local_T * local_T * local_T / 5 + 
                    nasa_coeffs[i * 15 + 6] / local_T) * GAS_CANSTANT * local_T / molecular_weights[i] * mass_fraction[i * num_cells + index];
        } else {
            h += (nasa_coeffs[i * 15 + 8] + nasa_coeffs[i * 15 + 9] * local_T / 2 + nasa_coeffs[i * 15 + 10] * local_T * local_T / 3 + 
                    nasa_coeffs[i * 15 + 11] * local_T * local_T * local_T / 4 + nasa_coeffs[i * 15 + 12] * local_T * local_T * local_T * local_T / 5 + 
                    nasa_coeffs[i * 15 + 13] / local_T) * GAS_CANSTANT * local_T / molecular_weights[i] * mass_fraction[i * num_cells + index];
        }
    }
    return h;
}

__global__ void calculate_temperature_kernel(int num_cells, int num_species, 
        const double *T_init, const double *h_target, 
        const double *nasa_coeffs, const double *molecular_weights, const double *mass_fraction,
        double *T_est, double atol, double rtol, int max_iter)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num_cells)
        return;

    double local_T = T_init[index];
    double local_h_target = h_target[index];
    double h, cp, delta_T;
    for (int n = 0; n < max_iter; ++n) {
        h = calculate_enthalpy_device_kernel(num_cells, num_species, index, local_T, nasa_coeffs, molecular_weights, mass_fraction);
        cp = calculate_cp_device_kernel(num_cells, num_species, index, nasa_coeffs, local_T, mass_fraction, molecular_weights);
        delta_T = (h - local_h_target) / cp;
        local_T -= delta_T;
        if (fabs(h - local_h_target) < atol || fabs(delta_T / local_T) < rtol) {
            break;
        }
    }
    T_est[index] = local_T;
}

void dfThermo::sync()
{
    checkCudaErrors(cudaStreamSynchronize(stream));
}

void dfThermo::setMassFraction(const double *d_mass_fraction)
{
    size_t threads_per_block = 1024;
    size_t blocks_per_grid = (num_cells + threads_per_block - 1) / threads_per_block;
    get_mole_fraction_mean_mole_weight<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_cells, num_species, 
            d_mass_fraction, d_molecular_weights, d_mole_fraction, d_mean_mole_weight);
}

void dfThermo::calculateTPolyGPU(const double *T)
{
    size_t threads_per_block = 1024;
    size_t blocks_per_grid = (num_cells + threads_per_block - 1) / threads_per_block;
    calculate_TPoly_kernel<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_cells, T, d_T_poly);
}

void dfThermo::calculatePsiGPU(const double *T, double *d_psi)
{
    size_t threads_per_block = 1024;
    size_t blocks_per_grid = (num_cells + threads_per_block - 1) / threads_per_block;
    calculate_psi_kernel<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_cells, T, d_mean_mole_weight, d_psi);
}

void dfThermo::calculateRhoGPU(const double *p, const double *psi, double *rho)
{
    size_t threads_per_block = 1024;
    size_t blocks_per_grid = (num_cells + threads_per_block - 1) / threads_per_block;
    calculate_rho_kernel<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_cells, p, psi, rho);
}

void dfThermo::calculateViscosityGPU(const double *T, double *viscosity)
{
    calculateTPolyGPU(T);

    size_t threads_per_block = 1024;
    size_t blocks_per_grid = (num_cells + threads_per_block - 1) / threads_per_block;
    calculate_viscosity_kernel<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_cells, num_species, 
            d_viscosity_coeffs, d_T_poly, T, d_mole_fraction, d_molecular_weights, d_species_viscosities, viscosity);
}

void dfThermo::calculateThermoConductivityGPU(const double *T, const double *d_mass_fraction, double *thermal_conductivity)
{
    size_t threads_per_block = 1024;
    size_t blocks_per_grid = (num_cells + threads_per_block - 1) / threads_per_block;
    calculate_thermoConductivity_kernel<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_cells, num_species, 
            d_nasa_coeffs, d_mass_fraction, d_thermal_conductivity_coeffs, d_T_poly, T, d_mole_fraction, d_molecular_weights, 
            d_species_thermal_conductivities, thermal_conductivity);
}

void dfThermo::calculateEnthalpyGPU(const double *T, double *enthalpy)
{
    size_t threads_per_block = 1024;
    size_t blocks_per_grid = (num_cells + threads_per_block - 1) / threads_per_block;
    calculate_enthalpy_kernel<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_cells, num_species, T, 
            d_nasa_coeffs, d_molecular_weights, d_mass_fraction, enthalpy);
}

void dfThermo::calculateTemperatureGPU(const double *T_init, const double *target_h, double *T, const double *d_mass_fraction, double atol, 
            double rtol, int max_iter)
{
    size_t threads_per_block = 1024;
    size_t blocks_per_grid = (num_cells + threads_per_block - 1) / threads_per_block;
    calculate_temperature_kernel<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_cells, num_species, 
            T_init, target_h, d_nasa_coeffs, d_molecular_weights, d_mass_fraction, T, atol, rtol, max_iter);
}