#include "dfThermo.h"
#include <filesystem>
#include <cmath>
#include <numeric>
#include <cassert>
#include <cstring>

#define PRINT_VECTOR(v) \
    for (auto i : v) \
        printf("%.8e ", i); \
    printf("\n");

#define PRINT_2D_VECTOR(v) \
    for (auto i : v) \
    { \
        for (auto j : i) \
            printf("%.8e ", j); \
        printf("\n");\
    }

using namespace std;

dfThermo::dfThermo(string mechanism_file, int num_cells) : mechanism_file(mechanism_file), num_cells(num_cells)
{
    // get thermo_coeff_file from mechanism_file
    string prefix = "thermo_";
    string suffix = ".txt";
    std::string baseName = std::filesystem::path(mechanism_file).stem().string();
    thermo_coeff_file = prefix + baseName + suffix;

    // check if thermo_coeff_file exists
    if (!std::filesystem::exists(thermo_coeff_file))
    {
        cout << "Thermo coefficient file does not exist!" << endl;
        exit(1);
    }

    // initialize coeffcients from thermo_coeff_file
    /*
    ifstream inputf(thermo_coeff_file, ifstream::in);
    if (!inputf.good())
    {
        cout << "Error opening file!" << endl;
        exit(1);
    }
    inputf >> num_species;

    molecular_weights.resize(num_species);
    for (int i = 0; i < num_species; i++) {
        inputf >> molecular_weights[i];
    }*/

    // read binary file
    FILE *fp = NULL;
    char *c_thermo_file = new char[thermo_coeff_file.length() + 1];
    strcpy(c_thermo_file, thermo_coeff_file.c_str());

    fp = fopen(c_thermo_file, "rb+");
    if (fp == NULL) {
        fprintf(stderr, "Failed to open input file: %s!\n", c_thermo_file);
        exit(EXIT_FAILURE);
    }

    fread(&num_species, sizeof(int), 1, fp);

    molecular_weights.resize(num_species);
    fread(molecular_weights.data(), sizeof(double), num_species, fp);

    species_names.resize(num_species);
    mass_fraction.resize(num_species);
    mole_fraction.resize(num_species);

    initCoeffsfromBinaryFile(fp);

    T_poly.resize(5);

    /*
    // initialize device data
    checkCudaErrors(cudaMalloc((void**)&d_nasa_coeffs, sizeof(double) * num_species * 15));
    checkCudaErrors(cudaMalloc((void**)&d_viscosity_coeffs, sizeof(double) * num_species * 5));
    checkCudaErrors(cudaMalloc((void**)&d_thermal_conductivity_coeffs, sizeof(double) * num_species * 5));
    checkCudaErrors(cudaMalloc((void**)&d_binary_diffusion_coeffs, sizeof(double) * num_species * num_species * 5));
    checkCudaErrors(cudaMalloc((void**)&d_molecular_weights, sizeof(double) * num_species));

    // copy data from host to device
    for (int i = 0; i < num_species; i++) {
        checkCudaErrors(cudaMemcpy(d_nasa_coeffs + i * 15, nasa_coeffs[i].data(), sizeof(double) * 15, cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(d_viscosity_coeffs + i * 5, viscosity_coeffs[i].data(), sizeof(double) * 5, cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(d_thermal_conductivity_coeffs + i * 5, thermal_conductivity_coeffs[i].data(), sizeof(double) * 5, cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(d_binary_diffusion_coeffs + i * 5, binary_diffusion_coeffs[i].data(), sizeof(double) * 5 * num_species, cudaMemcpyHostToDevice));
    }
    checkCudaErrors(cudaMemcpy(d_molecular_weights, molecular_weights.data(), sizeof(double) * num_species, cudaMemcpyHostToDevice));*/

    // constant memory
    // printf("d_nasa_coeffs ptr = %p\n", d_nasa_coeffs);
    // printf("nasa_coeffs ptr = %p\n", nasa_coeffs.data());
    // for (int i = 0; i < 7; i++) {
    //     printf("species = %d\n", i);
    //     checkCudaErrors(cudaMemcpyToSymbol(d_nasa_coeffs + i * 15, nasa_coeffs[i].data(), sizeof(double) * 15));
    //     checkCudaErrors(cudaMemcpyToSymbol(d_viscosity_coeffs + i * 5, viscosity_coeffs[i].data(), sizeof(double) * 5));
    //     checkCudaErrors(cudaMemcpyToSymbol(d_thermal_conductivity_coeffs + i * 5, thermal_conductivity_coeffs[i].data(), sizeof(double) * 5));
    //     checkCudaErrors(cudaMemcpyToSymbol(d_binary_diffusion_coeffs + i * 5, binary_diffusion_coeffs[i].data(), sizeof(double) * 5 * 7));
    // }
    // checkCudaErrors(cudaMemcpyToSymbol(d_molecular_weights, molecular_weights.data(), sizeof(double) * 7));
    checkCudaErrors(cudaStreamCreate(&stream));
    checkCudaErrors(cudaMalloc((void**)&d_mass_fraction, sizeof(double) * num_species * num_cells));
    checkCudaErrors(cudaMalloc((void**)&d_mole_fraction, sizeof(double) * num_species * num_cells));
    checkCudaErrors(cudaMalloc((void**)&d_mean_mole_weight, sizeof(double) * num_cells));
    checkCudaErrors(cudaMalloc((void**)&d_T_poly, sizeof(double) * 5 * num_cells));

    checkCudaErrors(cudaMalloc((void**)&d_species_viscosities, sizeof(double) * num_species * num_cells));
    checkCudaErrors(cudaMalloc((void**)&d_species_thermal_conductivities, sizeof(double) * num_species * num_cells));
    std::cout << "dfThermo initialized" << std::endl;
}

void dfThermo::readCoeffs(ifstream& inputf, int dimension, vector<vector<double>>& coeffs)
{
    coeffs.resize(num_species);
    for (int i = 0; i < num_species; i++) {
        coeffs[i].resize(dimension);
        for (int j = 0; j < dimension; j++) {
            inputf >> coeffs[i][j];   
        }
    }
}

void dfThermo::readCoeffsBinary(FILE* fp, int dimension, vector<vector<double>>& coeffs)
{
    coeffs.resize(num_species);
    for (int i = 0; i < num_species; i++) {
        coeffs[i].resize(dimension);
        fread(coeffs[i].data(), sizeof(double), dimension, fp);
    }
}

void dfThermo::initCoeffs(ifstream& inputf)
{
    readCoeffs(inputf, 15, nasa_coeffs);
    readCoeffs(inputf, 5, viscosity_coeffs);
    readCoeffs(inputf, 5, thermal_conductivity_coeffs);
    readCoeffs(inputf, num_species * 5, binary_diffusion_coeffs);
}

void dfThermo::initCoeffsfromBinaryFile(FILE* fp)
{
    readCoeffsBinary(fp, 15, nasa_coeffs);
    readCoeffsBinary(fp, 5, viscosity_coeffs);
    readCoeffsBinary(fp, 5, thermal_conductivity_coeffs);
    readCoeffsBinary(fp, num_species * 5, binary_diffusion_coeffs);
}

void dfThermo::setMassFraction(vector<double>& mass_fraction)
{
    // assert sum of mass fraction is 1
    assert(mass_fraction.size() == num_species);
    assert((std::accumulate(mass_fraction.begin(), mass_fraction.end(), 0.0) - 1.0) < 1e-10);

    this->mass_fraction = mass_fraction;
    double sum = 0.;
    for (int i = 0; i < num_species; ++i) {
        sum += mass_fraction[i] / molecular_weights[i];
    }
    meanMolecularWeight = 0.;
    for (int i = 0; i < num_species; ++i) {
        mole_fraction[i] = mass_fraction[i] / (molecular_weights[i] * sum);
        meanMolecularWeight += mole_fraction[i] * molecular_weights[i];
    }
}

void dfThermo::calculateTPoly(double T)
{
    T_poly[0] = 1.0;
    T_poly[1] = log(T);
    T_poly[2] = T_poly[1] * T_poly[1];
    T_poly[3] = T_poly[1] * T_poly[2];
    T_poly[4] = T_poly[2] * T_poly[2];
}

double dfThermo::calculatePsi(double T)
{
    return meanMolecularWeight / (GAS_CANSTANT * T);
}

double dfThermo::calculateRho(double p, double psi)
{
    return p * psi;
}

double dfThermo::calculateViscosity(double T)
{
    calculateTPoly(T);
    std::vector<double> species_viscosities(num_species);
    
    double dot_product;
    for (int i = 0; i < num_species; i++) {
        dot_product = 0.;
        for (int j = 0; j < 5; j++) {
            dot_product += viscosity_coeffs[i][j] * T_poly[j];
        }
        species_viscosities[i] = (dot_product * dot_product) * sqrt(T);
    }
    
    double mu_mix = 0.;
    for (int i = 0; i < num_species; i++) {
        double temp = 0.;
        for (int j = 0; j < num_species; j++) {
            temp += mole_fraction[j] / pow(8, 0.5) *
            pow((1 + molecular_weights[i] / molecular_weights[j]), -0.5) *
            pow(1.0 + sqrt(species_viscosities[i] / species_viscosities[j]) *
            pow(molecular_weights[j] / molecular_weights[i], 0.25), 2.0);
        }
        mu_mix += mole_fraction[i] * species_viscosities[i] / temp;
    }
    return mu_mix;
}

double dfThermo::calculateThermoConductivity(double T)
{
    vector<double> species_thermal_conductivities(num_species);

    double dot_product;
    for (int i = 0; i < num_species; i++) {
        dot_product = 0.;
        for (int j = 0; j < 5; j++) {
            dot_product += thermal_conductivity_coeffs[i][j] * T_poly[j];
        }
        species_thermal_conductivities[i] = dot_product * sqrt(T);
    }

    double sum_conductivity = 0.;
    double sum_inv_conductivity = 0.;

    for (int i = 0; i < num_species; ++i) {
        sum_conductivity += mole_fraction[i] * species_thermal_conductivities[i];
        sum_inv_conductivity += mole_fraction[i] / species_thermal_conductivities[i];
    }

    double lambda_mix = 0.5 * (sum_conductivity + 1.0 / sum_inv_conductivity);
    double cp = calculateCp(T);
    return lambda_mix / cp;
}

double dfThermo::calculateEnthalpy(double T)
{
    double h = 0.;
    double term1, term2;
    for (int i = 0; i < num_species; i++) {
        if (T > nasa_coeffs[i][0]) {
            term1 = nasa_coeffs[i][1] + nasa_coeffs[i][2] * T / 2 + nasa_coeffs[i][3] * T * T / 3 + nasa_coeffs[i][4] * T * T * T / 4 + nasa_coeffs[i][5] * T * T * T * T / 5 + nasa_coeffs[i][6] / T;
            term2 = GAS_CANSTANT * T / molecular_weights[i];
            h += mass_fraction[i] * term1 * term2;
        } else {
            term1 = nasa_coeffs[i][8] + nasa_coeffs[i][9] * T / 2 + nasa_coeffs[i][10] * T * T / 3 + nasa_coeffs[i][11] * T * T * T / 4 + nasa_coeffs[i][12] * T * T * T * T / 5 + nasa_coeffs[i][13] / T;
            term2 = GAS_CANSTANT * T / molecular_weights[i];
            h += mass_fraction[i] * term1 * term2;
        }
    }
    return h;
}

double dfThermo::calculateCp(double T)
{
    double cp = 0.;

    for (int i = 0; i < num_species; i++) {
        if (T > nasa_coeffs[i][0]) {
            cp += mass_fraction[i] * (nasa_coeffs[i][1] + nasa_coeffs[i][2] * T + nasa_coeffs[i][3] * T * T + nasa_coeffs[i][4] * T * T * T + nasa_coeffs[i][5] * T * T * T * T) * GAS_CANSTANT / molecular_weights[i];
        } else {
            cp += mass_fraction[i] * (nasa_coeffs[i][8] + nasa_coeffs[i][9] * T + nasa_coeffs[i][10] * T * T + nasa_coeffs[i][11] * T * T * T + nasa_coeffs[i][12] * T * T * T * T) * GAS_CANSTANT / molecular_weights[i];
        }
    }
    return cp;
}

double dfThermo::calculateTemperature(const double T_init, const double h_target, 
        double atol, double rtol, int max_iter)
{
    double T = T_init + 10.;
    for (int n = 0; n < max_iter; ++n) {
        double h = calculateEnthalpy(T);
        double cp = calculateCp(T);
        double delta_h = h - h_target;
        double delta_T = delta_h / cp;

        T -= delta_T;

        if (fabs(delta_h) < atol || fabs(delta_T / T) < rtol) {
            // cout << "Convergence achieved within " << n << " steps" << endl;
            return T;
        }
    }

    std::cerr << "Convergence not achieved within " << max_iter << " steps" << std::endl;
    return T; // Return the current temperature as the best estimate
}

dfThermo::~dfThermo()
{
    // destructor
}