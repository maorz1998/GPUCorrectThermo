#include "dfThermo.h"
#include <filesystem>
#include <cmath>

using namespace std;

dfThermo::dfThermo(string mechanism_file) : mechanism_file(mechanism_file)
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
    ifstream inputf(thermo_coeff_file, ifstream::in);
    if (!inputf.good())
    {
        cout << "Error opening file!" << endl;
        exit(1);
    }
    inputf >> num_species;
    initCoeffs(inputf);

    T_poly.resize(5);
}

void dfThermo::readCoeffs(ifstream& inputf, int dimension, vector<vector<double>>& coeffs)
{
    coeffs.resize(dimension);
    for (int i = 0; i < num_species; i++) {
        for (int j = 0; j < dimension; j++) {
            inputf >> coeffs[i][j];   
        }
    }
}

void dfThermo::initCoeffs(ifstream& inputf)
{
    readCoeffs(inputf, 15, nasa_coeffs);
    readCoeffs(inputf, 5, viscosity_coeffs);
    readCoeffs(inputf, 5, thermal_conductivity_coeffs);
    readCoeffs(inputf, num_species * 5, binary_diffusion_coeffs);
}

void dfThermo::calculateTPoly(double T)
{
    T_poly[0] = 1.0;
    T_poly[1] = log(T);
    T_poly[2] = T_poly[1] * T_poly[1];
    T_poly[3] = T_poly[1] * T_poly[2];
    T_poly[4] = T_poly[2] * T_poly[2];
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
            temp += pow(mole_fraction[j] / 8, 0.5) * pow((1 + molecular_weights[i] / molecular_weights[j]), -0.5) * 
                pow(1 + (species_viscosities[i] / species_viscosities[j]), 0.5) * 
                pow((molecular_weights[j] / molecular_weights[i]), 0.25);
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
    return lambda_mix;
}

double dfThermo::calculateEnthalpy(double T, double *Y)
{
    double h = 0.;
    double term1, term2;
    for (int i = 0; i < num_species; i++) {
        if (T > nasa_coeffs[i][0]) {
            term1 = nasa_coeffs[i][1] + nasa_coeffs[i][2] * T / 2 + nasa_coeffs[i][3] * T * T / 3 + nasa_coeffs[i][4] * T * T * T / 4 + nasa_coeffs[i][5] * T * T * T * T / 5 + nasa_coeffs[i][6] / T;
            term2 = gas_constant * T / molecular_weights[i];
            h += Y[i] * term1 * term2;
        }
        else {
            term1 = nasa_coeffs[i][8] + nasa_coeffs[i][9] * T / 2 + nasa_coeffs[i][10] * T * T / 3 + nasa_coeffs[i][11] * T * T * T / 4 + nasa_coeffs[i][12] * T * T * T * T / 5 + nasa_coeffs[i][13] / T;
            term2 = gas_constant * T / molecular_weights[i];
            h += Y[i] * term1 * term2;
        }
    }
    return h;
}

double dfThermo::calculateCp(double T, double *Y)
{
    double cp = 0.;

    for (int i = 0; i < num_species; i++) {
        if (T > nasa_coeffs[i][0]) {
            cp += Y[i] * (nasa_coeffs[i][1] + nasa_coeffs[i][2] * T + nasa_coeffs[i][3] * T * T + nasa_coeffs[i][4] * T * T * T + nasa_coeffs[i][5] * T * T * T * T) * gas_constant / molecular_weights[i];
        } else {
            cp += Y[i] * (nasa_coeffs[i][8] + nasa_coeffs[i][9] * T + nasa_coeffs[i][10] * T * T + nasa_coeffs[i][11] * T * T * T + nasa_coeffs[i][12] * T * T * T * T) * gas_constant / molecular_weights[i];
        }
    }
    return cp;
}

double dfThermo::calculateTemperature(const double T_init, const double h_target, double *Y, 
        double atol, double rtol, int max_iter)
{
    double T = T_init + 10.;
    for (int n = 0; n < max_iter; ++n) {
        double h = calculateEnthalpy(T, Y);
        double cp = calculateCp(T, Y);
        double delta_h = h - h_target;
        double delta_T = delta_h / cp;

        T -= delta_T;

        if (fabs(delta_h) < atol || fabs(delta_T / T) < rtol) {
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