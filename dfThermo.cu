#include "dfThermo.h"
#include <filesystem>
#include <cmath>
#include <numeric>
#include <cassert>
#include <cstring>
#include "device_launch_parameters.h"

#define TICK_INIT_EVENT \
    float time_elapsed_kernel=0;\
    cudaEvent_t start_kernel, stop_kernel;\
    checkCudaErrors(cudaEventCreate(&start_kernel));\
    checkCudaErrors(cudaEventCreate(&stop_kernel));

#define TICK_START_EVENT \
    checkCudaErrors(cudaEventRecord(start_kernel,0));

#define TICK_END_EVENT(prefix) \
    checkCudaErrors(cudaEventRecord(stop_kernel,0));\
    checkCudaErrors(cudaEventSynchronize(start_kernel));\
    checkCudaErrors(cudaEventSynchronize(stop_kernel));\
    checkCudaErrors(cudaEventElapsedTime(&time_elapsed_kernel,start_kernel,stop_kernel));\
    printf("try %s 执行时间：%lf(ms)\n", #prefix, time_elapsed_kernel);

#define SQRT8 2.8284271247461903

// use constant memory
__constant__ __device__ double d_nasa_coeffs[7*15];
__constant__ __device__ double d_viscosity_coeffs[7*5];
__constant__ __device__ double d_thermal_conductivity_coeffs[7*5];
__constant__ __device__ double d_binary_diffusion_coeffs[7*7*5];
__constant__ __device__ double d_molecular_weights[7];
__constant__ __device__ double d_viscosity_conatant1[7*7];
__constant__ __device__ double d_viscosity_conatant2[7*7];

void init_const_coeff_ptr(std::vector<std::vector<double>>& nasa_coeffs, std::vector<std::vector<double>>& viscosity_coeffs,
        std::vector<std::vector<double>>& thermal_conductivity_coeffs, std::vector<std::vector<double>>& binary_diffusion_coeffs,
        std::vector<double>& molecular_weights)
{
    double *d_tmp;
    checkCudaErrors(cudaMalloc((void**)&d_tmp, sizeof(double) * 7 * 15));
    double nasa_coeffs_tmp[7*15];
    double viscosity_coeffs_tmp[7*5];
    double thermal_conductivity_coeffs_tmp[7*5];
    double binary_diffusion_coeffs_tmp[7*7*5];
    double viscosity_conatant1_tmp[7*7];
    double viscosity_conatant2_tmp[7*7];

    for (int i = 0; i < 7; i++) {
        std::copy(nasa_coeffs[i].begin(), nasa_coeffs[i].end(), nasa_coeffs_tmp + i * 15);
        std::copy(viscosity_coeffs[i].begin(), viscosity_coeffs[i].end(), viscosity_coeffs_tmp + i * 5);
        std::copy(thermal_conductivity_coeffs[i].begin(), thermal_conductivity_coeffs[i].end(), thermal_conductivity_coeffs_tmp + i * 5);
        std::copy(binary_diffusion_coeffs[i].begin(), binary_diffusion_coeffs[i].end(), binary_diffusion_coeffs_tmp + i * 5 * 7);
        for (int j = 0; j < 7; j++) {
            viscosity_conatant1_tmp[i * 7 + j] = pow((1 + molecular_weights[i] / molecular_weights[j]), -0.5);
            viscosity_conatant2_tmp[i * 7 + j] = pow(molecular_weights[j] / molecular_weights[i], 0.25);
        }
    }
    checkCudaErrors(cudaMemcpyToSymbol(d_nasa_coeffs, nasa_coeffs_tmp, sizeof(double) * 15 * 7));
    checkCudaErrors(cudaMemcpyToSymbol(d_viscosity_coeffs, viscosity_coeffs_tmp, sizeof(double) * 5 * 7));
    checkCudaErrors(cudaMemcpyToSymbol(d_thermal_conductivity_coeffs, thermal_conductivity_coeffs_tmp, sizeof(double) * 5 * 7));
    checkCudaErrors(cudaMemcpyToSymbol(d_binary_diffusion_coeffs, binary_diffusion_coeffs_tmp, sizeof(double) * 5 * 7 * 7));
    checkCudaErrors(cudaMemcpyToSymbol(d_molecular_weights, molecular_weights.data(), sizeof(double) * 7));
    checkCudaErrors(cudaMemcpyToSymbol(d_viscosity_conatant1, viscosity_conatant1_tmp, sizeof(double) * 7 * 7));
    checkCudaErrors(cudaMemcpyToSymbol(d_viscosity_conatant2, viscosity_conatant2_tmp, sizeof(double) * 7 * 7));
}

__global__ void warmup(int num_cells)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num_cells)
        return;
}

__global__ void get_mole_fraction_mean_mole_weight(int num_cells, int num_species, const double *d_Y, 
        double *mole_fraction, double *mean_mole_weight)
{   
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num_cells)
        return;
    
    double sum = 0.;
    for (int i = 0; i < num_species; i++) {
        sum += d_Y[i * num_cells + index] / d_molecular_weights[i];
    }
    double meanMoleWeight = 0.;
    for (int i = 0; i < num_species; i++) {
        mole_fraction[i * num_cells + index] = d_Y[i * num_cells + index] / (d_molecular_weights[i] * sum);
        meanMoleWeight += mole_fraction[i * num_cells + index] * d_molecular_weights[i];
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

__global__ void calculate_viscosity_kernel(int num_cells, int num_species,
        const double *T_poly, const double *T, const double *mole_fraction,
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
            dot_product += d_viscosity_coeffs[i * 5 + j] * T_poly[num_cells * j + index];
        }
        species_viscosities[i * num_cells + index] = (dot_product * dot_product) * sqrt(local_T);
    }

    double mu_mix = 0.;
    for (int i = 0; i < num_species; i++) {
        double temp = 0.;
        for (int j = 0; j < num_species; j++) {
            temp += mole_fraction[num_cells * j + index] / SQRT8 *
            d_viscosity_conatant1[i * 7 + j] * // constant 1
            pow(1.0 + sqrt(species_viscosities[i * num_cells + index] / species_viscosities[j * num_cells + index]) *
            d_viscosity_conatant2[i * 7 + j], 2.0); // constant 2
        }
        mu_mix += mole_fraction[num_cells * i + index] * species_viscosities[i * num_cells + index] / temp;
    }

    mu[index] = mu_mix;
}

__device__ double calculate_cp_device_kernel(int num_cells, int num_species, int index, 
        const double local_T, const double *mass_fraction)
{   
    double cp = 0.;

    for (int i = 0; i < num_species; i++) {
        if (local_T > d_nasa_coeffs[i * 15 + 0]) {
            cp += mass_fraction[i * num_cells + index] * (d_nasa_coeffs[i * 15 + 1] + d_nasa_coeffs[i * 15 + 2] * local_T + d_nasa_coeffs[i * 15 + 3] * local_T * local_T + 
                    d_nasa_coeffs[i * 15 + 4] * local_T * local_T * local_T + 
                    d_nasa_coeffs[i * 15 + 5] * local_T * local_T * local_T * local_T) * GAS_CANSTANT / d_molecular_weights[i];
        } else {
            cp += mass_fraction[i * num_cells + index] * (d_nasa_coeffs[i * 15 + 8] + d_nasa_coeffs[i * 15 + 9] * local_T + d_nasa_coeffs[i * 15 + 10] * local_T * local_T + 
                    d_nasa_coeffs[i * 15 + 11] * local_T * local_T * local_T + 
                    d_nasa_coeffs[i * 15 + 12] * local_T * local_T * local_T * local_T) * GAS_CANSTANT / d_molecular_weights[i];
        }
    }
    return cp;
}

__global__ void calculate_thermoConductivity_kernel(int num_cells, int num_species, 
        const double *nasa_coeffs, const double *mass_fraction,
        const double *T_poly, const double *T, const double *mole_fraction,
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
            dot_product += d_thermal_conductivity_coeffs[i * 5 + j] * T_poly[num_cells * j + index];
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

    double cp = calculate_cp_device_kernel(num_cells, num_species, index, local_T, mass_fraction);

    thermal_conductivity[index] = lambda_mix / cp;
}

__device__ double calculate_enthalpy_device_kernel(int num_cells, int num_species, int index, const double local_T,
        const double *mass_fraction)
{
    double h = 0.;

    for (int i = 0; i < num_species; i++) {
        if (local_T > d_nasa_coeffs[i * 15 + 0]) {
            h += (d_nasa_coeffs[i * 15 + 1] + d_nasa_coeffs[i * 15 + 2] * local_T / 2 + d_nasa_coeffs[i * 15 + 3] * local_T * local_T / 3 + 
                    d_nasa_coeffs[i * 15 + 4] * local_T * local_T * local_T / 4 + d_nasa_coeffs[i * 15 + 5] * local_T * local_T * local_T * local_T / 5 + 
                    d_nasa_coeffs[i * 15 + 6] / local_T) * GAS_CANSTANT * local_T / d_molecular_weights[i] * mass_fraction[i * num_cells + index];
        } else {
            h += (d_nasa_coeffs[i * 15 + 8] + d_nasa_coeffs[i * 15 + 9] * local_T / 2 + d_nasa_coeffs[i * 15 + 10] * local_T * local_T / 3 + 
                    d_nasa_coeffs[i * 15 + 11] * local_T * local_T * local_T / 4 + d_nasa_coeffs[i * 15 + 12] * local_T * local_T * local_T * local_T / 5 + 
                    d_nasa_coeffs[i * 15 + 13] / local_T) * GAS_CANSTANT * local_T / d_molecular_weights[i] * mass_fraction[i * num_cells + index];
        }
    }
    return h;
}

__global__ void calculate_temperature_kernel(int num_cells, int num_species, 
        const double *T_init, const double *h_target, const double *mass_fraction,
        double *T_est, double atol, double rtol, int max_iter)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num_cells)
        return;

    double local_T = T_init[index];
    double local_h_target = h_target[index];
    double h, cp, delta_T;
    for (int n = 0; n < max_iter; ++n) {
        h = calculate_enthalpy_device_kernel(num_cells, num_species, index, local_T, mass_fraction);
        cp = calculate_cp_device_kernel(num_cells, num_species, index, local_T, mass_fraction);
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
    TICK_INIT_EVENT;
    size_t threads_per_block = 1024;
    size_t blocks_per_grid = (num_cells + threads_per_block - 1) / threads_per_block;
    printf("num_cells = %d\n", num_cells);
    printf("warm up ...\n");
    warmup<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_cells);

    TICK_START_EVENT;
    get_mole_fraction_mean_mole_weight<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_cells, num_species, 
            d_mass_fraction, d_mole_fraction, d_mean_mole_weight);
    TICK_END_EVENT(get_mole_fraction_mean_mole_weight);
}

void dfThermo::calculateTPolyGPU(const double *T)
{
    TICK_INIT_EVENT;
    size_t threads_per_block = 1024;
    size_t blocks_per_grid = (num_cells + threads_per_block - 1) / threads_per_block;
    TICK_START_EVENT;
    calculate_TPoly_kernel<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_cells, T, d_T_poly);
    TICK_END_EVENT(calculate_TPoly_kernel);
}

void dfThermo::calculatePsiGPU(const double *T, double *d_psi)
{
    TICK_INIT_EVENT;
    size_t threads_per_block = 1024;
    size_t blocks_per_grid = (num_cells + threads_per_block - 1) / threads_per_block;
    TICK_START_EVENT;
    calculate_psi_kernel<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_cells, T, d_mean_mole_weight, d_psi);
    TICK_END_EVENT(calculate_psi_kernel);
}

void dfThermo::calculateRhoGPU(const double *p, const double *psi, double *rho)
{
    TICK_INIT_EVENT;
    size_t threads_per_block = 1024;
    size_t blocks_per_grid = (num_cells + threads_per_block - 1) / threads_per_block;
    TICK_START_EVENT;
    calculate_rho_kernel<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_cells, p, psi, rho);
    TICK_END_EVENT(calculate_rho_kernel);
}

void dfThermo::calculateViscosityGPU(const double *T, double *viscosity)
{
    calculateTPolyGPU(T);

    TICK_INIT_EVENT;
    size_t threads_per_block = 1024;
    size_t blocks_per_grid = (num_cells + threads_per_block - 1) / threads_per_block;
    TICK_START_EVENT;
    calculate_viscosity_kernel<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_cells, num_species, 
            d_T_poly, T, d_mole_fraction, d_species_viscosities, viscosity);
    TICK_END_EVENT(calculate_viscosity_kernel);
}

void dfThermo::calculateThermoConductivityGPU(const double *T, const double *d_mass_fraction, double *thermal_conductivity)
{
    TICK_INIT_EVENT;
    size_t threads_per_block = 1024;
    size_t blocks_per_grid = (num_cells + threads_per_block - 1) / threads_per_block;
    TICK_START_EVENT;
    calculate_thermoConductivity_kernel<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_cells, num_species, 
            d_nasa_coeffs, d_mass_fraction, d_T_poly, T, d_mole_fraction,
            d_species_thermal_conductivities, thermal_conductivity);
    TICK_END_EVENT(calculate_thermoConductivity_kernel);
}

void dfThermo::calculateTemperatureGPU(const double *T_init, const double *target_h, double *T, const double *d_mass_fraction, double atol, 
            double rtol, int max_iter)
{
    TICK_INIT_EVENT;
    size_t threads_per_block = 1024;
    size_t blocks_per_grid = (num_cells + threads_per_block - 1) / threads_per_block;

    TICK_START_EVENT;
    calculate_temperature_kernel<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_cells, num_species, 
            T_init, target_h, d_mass_fraction, T, atol, rtol, max_iter);
    TICK_END_EVENT(calculate_temperature_kernel);
}

void dfThermo::compareThermoConductivity(const double *d_thermal_conductivity, const double *thermal_conductivity, 
        bool printFlag)
{
    std::vector<double> h_thermal_conductivity(num_cells);
    checkCudaErrors(cudaMemcpy(h_thermal_conductivity.data(), d_thermal_conductivity, sizeof(double) * num_cells, cudaMemcpyDeviceToHost));
    printf("compare thermal_conductivity\n");
    checkVectorEqual(num_cells, thermal_conductivity, h_thermal_conductivity.data(), 1e-14, printFlag);
}

void dfThermo::compareViscosity(const double *d_viscosity, const double *viscosity, bool printFlag)
{
    std::vector<double> h_viscosity(num_cells);
    checkCudaErrors(cudaMemcpy(h_viscosity.data(), d_viscosity, sizeof(double) * num_cells, cudaMemcpyDeviceToHost));
    printf("compare viscosity\n");
    checkVectorEqual(num_cells, viscosity, h_viscosity.data(), 1e-14, printFlag);
}