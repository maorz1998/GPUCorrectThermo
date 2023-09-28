#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <cmath>

#include <cuda_runtime.h>


static const char *_cudaGetErrorEnum(cudaError_t error) {
  return cudaGetErrorName(error);
}

template <typename T>
void check(T result, char const *const func, const char *const file,
        int const line) {
  if (result) {
    fprintf(stderr, "cuda error at %s:%d code=%d(%s) \"%s\" \n", file, line,
            static_cast<unsigned int>(result), _cudaGetErrorEnum(result), func);
    exit(EXIT_FAILURE);
  }
}

#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)

#define GAS_CANSTANT 8314.46261815324

void init_const_coeff_ptr(std::vector<std::vector<double>>& nasa_coeffs, std::vector<std::vector<double>>& viscosity_coeffs,
        std::vector<std::vector<double>>& thermal_conductivity_coeffs, std::vector<std::vector<double>>& binary_diffusion_coeffs,
        std::vector<double>& molecular_weights);

inline void checkVectorEqual(int count, const double* basevec, double* vec, double max_relative_error, bool print = false) {
    for (int i = 0; i < count; ++i)
    {
        double abs_diff = fabs(basevec[i] - vec[i]);
        double rel_diff = fabs(basevec[i] - vec[i]) / fabs(basevec[i]);
        if (print)
            fprintf(stderr, "index %d, cpu data: %.24lf, gpu data: %.24lf, relative error: %.24lf\n", i, basevec[i], vec[i], rel_diff);
        // if (abs_diff > 1e-12 && rel_diff > max_relative_error && !std::isinf(rel_diff))
        if (abs_diff > 1e-15 && rel_diff > max_relative_error)
            fprintf(stderr, "mismatch index %d, cpu data: %.30lf, gpu data: %.30lf, relative error: %.16lf\n", i, basevec[i], vec[i], rel_diff);
    }   
}

class dfThermo
{
    // private data members
    const std::string mechanism_file;
    std::string thermo_coeff_file;

    // private member functions
    void readCoeffs(std::ifstream& inputf, int dimension, std::vector<std::vector<double>>& coeffs);
    void readCoeffsBinary(FILE* fp, int dimension, std::vector<std::vector<double>>& coeffs);
    void initCoeffs(std::ifstream& inputf);
    void initCoeffsfromBinaryFile(FILE* fp);

public:
    // cuda resource
    cudaStream_t stream;

    // public data members
    int num_species;
    int num_cells;

    // constant value
    double gas_constant;

    // thermo coeffs
    std::vector<std::vector<double>> nasa_coeffs;
    std::vector<std::vector<double>> viscosity_coeffs;
    std::vector<std::vector<double>> thermal_conductivity_coeffs;
    std::vector<std::vector<double>> binary_diffusion_coeffs;
    std::vector<double> molecular_weights;

    // double *d_nasa_coeffs, *d_viscosity_coeffs, *d_thermal_conductivity_coeffs, *d_binary_diffusion_coeffs, *d_molecular_weights;


    // species info
    std::vector<std::string> species_names;
    std::vector<double> mass_fraction;
    std::vector<double> mole_fraction;
    double meanMolecularWeight;

    double *d_mass_fraction, *d_mole_fraction, *d_mean_mole_weight;

    // intermediate variables
    std::vector<double> T_poly;

    double *d_T_poly;
    double *d_species_viscosities, *d_species_thermal_conductivities;

    // constructor
    dfThermo(std::string mechanism_file, int num_cells = 1);

    // destructor
    ~dfThermo();

    // public member functions

    // set mass fraction
    void setMassFraction(std::vector<double>& mass_fraction);
    void setMassFraction(const double *d_mass_fraction);

    // *** CPU functions ***
    // calculateTPoly
    void calculateTPoly(double T);
    // calculateViscosity must be called earlier than calculateThermoConductivity to calculate T_poly
    double calculatePsi(double T);
    double calculateRho(double p, double psi);
    double calculateViscosity(double T);
    double calculateThermoConductivity(double T);
    double calculateEnthalpy(double T);
    double calculateCp(double T);
    double calculateTemperature(double T_init, double target_h, double atol = 1e-7, 
            double rtol = 1e-7, int max_iter = 20);

    // *** GPU functions ***
    void calculateTPolyGPU(const double *T);
    void calculatePsiGPU(const double *T, double *psi);
    void calculateRhoGPU(const double *p, const double *psi, double *rho);
    void calculateViscosityGPU(const double *T, double *viscosity);
    void calculateThermoConductivityGPU(const double *T, const double *d_mass_fraction, double *thermal_conductivity);
    void calculateEnthalpyGPU(const double *T, double *enthalpy);
    void calculateTemperatureGPU(const double *T_init, const double *target_h, double *T, const double *d_mass_fraction, 
            double atol = 1e-7, double rtol = 1e-7, int max_iter = 20);
    
    void compareThermoConductivity(const double *d_thermal_conductivity, const double *thermal_conductivity,
            bool printFlag);
    void compareViscosity(const double *d_viscosity, const double *viscosity, bool printFlag);

    void sync();

    // getter functions
};