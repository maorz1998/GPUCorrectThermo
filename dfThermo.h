#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>

#define GAS_CANSTANT 8314.46261815324


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
    // public data members
    int num_species;

    // constant value
    double gas_constant;

    // thermo coeffs
    std::vector<std::vector<double>> nasa_coeffs;
    std::vector<std::vector<double>> viscosity_coeffs;
    std::vector<std::vector<double>> thermal_conductivity_coeffs;
    std::vector<std::vector<double>> binary_diffusion_coeffs;
    std::vector<double> molecular_weights;

    // species info
    std::vector<std::string> species_names;
    std::vector<double> mass_fraction;
    std::vector<double> mole_fraction;


    // intermediate variables
    std::vector<double> T_poly;

    // constructor
    dfThermo(std::string mechanism_file);

    // destructor
    ~dfThermo();

    // public member functions

    // set mass fraction
    void setMassFraction(std::vector<double>& mass_fraction);

    // calculateTPoly
    void calculateTPoly(double T);
    // calculateViscosity must be called earlier than calculateThermoConductivity to calculate T_poly
    double calculateViscosity(double T);
    double calculateThermoConductivity(double T);
    double calculateEnthalpy(double T);
    double calculateCp(double T);
    double calculateTemperature(double T_init, double target_h, double atol = 1e-7, 
            double rtol = 1e-7, int max_iter = 20);

    // getter functions
};